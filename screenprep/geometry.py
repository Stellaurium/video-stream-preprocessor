from __future__ import annotations
import copy
import json
from pathlib import Path
import cv2
import numpy as np

MAX_PIXELS = 16_777_216
INTERPOLATIONS = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4}
DEFAULT_INTERPOLATION = "lanczos4"
DEFAULT_SHARPEN_AMOUNT = 0.2

def checked_size(size):
    if len(size) != 2 or any(isinstance(v, bool) or int(v) != v for v in size):
        raise ValueError("画面尺寸必须为两个整数")
    w, h = map(int, size)
    if min(w, h) < 2 or max(w, h) > 8192 or w * h > MAX_PIXELS:
        raise ValueError("画面尺寸超出范围（每边 2–8192，最多 1600 万像素）")
    return w, h

def quad(points, size, require_inside=True):
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("需要四个有效角点")
    w, h = size
    if require_inside and ((p[:, 0] < -2).any() or (p[:, 0] > w + 1).any() or (p[:, 1] < -2).any() or (p[:, 1] > h + 1).any()):
        raise ValueError("四个角点必须都位于当前画面内；请撤销裁剪或显示完整屏幕")
    # Ordered TL, TR, BR, BL; require convex clockwise polygon in screen coordinates.
    edges = np.roll(p, -1, axis=0) - p
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if (cross <= 0.1).any() or cv2.contourArea(p.astype(np.float32)) < 25:
        raise ValueError("请按左上 → 右上 → 右下 → 左下选取四点，四边形不能交叉或过小")
    return p

def rectangle(size):
    w,h = size
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float64)


def project_polygon(points,matrix):
    points = np.asarray(points,np.float64).reshape(-1,2)
    homogeneous = np.c_[points,np.ones(len(points))] @ np.asarray(matrix,np.float64).T
    denominator = homogeneous[:,2]
    scale = max(float(np.max(np.abs(denominator))),1e-15)
    if not np.isfinite(homogeneous).all() or np.min(np.abs(denominator))<scale*1e-9 or np.min(denominator)*np.max(denominator)<=0:
        raise ValueError("所选输出范围跨过透视无穷远线，无法完整展开。请选择较小的留边比例，或先裁剪远处背景。")
    projected = homogeneous[:,:2]/denominator[:,None]
    if np.max(np.abs(projected))>1e9:
        raise ValueError("透视延伸范围过大，请缩小留边或先裁剪背景")
    return projected


class Pipeline:
    """Immutable-at-render snapshots; all projective steps merge into one warp.

    Each step is restricted to the previous image domain. This prevents a later
    correction from restoring pixels excluded by an earlier crop.
    """
    def __init__(self, input_size, interpolation=DEFAULT_INTERPOLATION, sharpen_amount=DEFAULT_SHARPEN_AMOUNT):
        self.input_size = checked_size(input_size)
        self.set_render_options(interpolation,sharpen_amount)
        self.operations = []
        self.redo_stack = []
        self._mask_cache = None

    def set_render_options(self,interpolation,sharpen_amount):
        if not isinstance(interpolation,str) or interpolation not in INTERPOLATIONS:
            raise ValueError("未知的图像采样方式")
        if isinstance(sharpen_amount,bool) or not isinstance(sharpen_amount,(int,float)) or not np.isfinite(sharpen_amount) or not 0<=sharpen_amount<=0.6:
            raise ValueError("轻锐化强度必须在 0%–60% 之间")
        self.interpolation = interpolation
        self.sharpen_amount = float(sharpen_amount)

    @property
    def render_options(self):
        return dict(interpolation=self.interpolation,sharpen_amount=self.sharpen_amount)

    @property
    def output_size(self):
        return tuple(self.operations[-1]["output_size"]) if self.operations else self.input_size

    @property
    def matrix(self):
        total = np.eye(3, dtype=np.float64)
        for op in self.operations:
            total = np.asarray(op["matrix"], dtype=np.float64) @ total
        return total / total[2, 2] if abs(total[2, 2]) > 1e-12 else total

    def clone(self):
        result = Pipeline(self.input_size,**self.render_options)
        result.operations = copy.deepcopy(self.operations)
        result.redo_stack = copy.deepcopy(self.redo_stack)
        return result

    def crop(self, rect):
        if len(rect) != 4 or not np.isfinite(rect).all():
            raise ValueError("裁剪区域无效")
        if any(int(v) != v for v in rect):
            raise ValueError("裁剪边界必须为整数像素")
        x, y, w, h = map(int, rect)
        checked_size((w, h))
        iw, ih = self.output_size
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            raise ValueError("裁剪范围超出当前画面")
        self._append("crop", [[1, 0, -x], [0, 1, -y], [0, 0, 1]], (w, h), rect=[x, y, w, h])

    def perspective(self, points, output_size, label="四点校正", details=None, *, framing="screen", margin_ratio=0.0):
        reference_size = checked_size(output_size)
        p = quad(points, self.output_size)
        if framing not in ("screen","margin","canvas"):
            raise ValueError("未知的校正输出范围")
        if not np.isfinite(margin_ratio) or not 0<=margin_ratio<=2:
            raise ValueError("每侧留边比例必须在 0%–200% 之间")
        w,h = reference_size
        m = cv2.getPerspectiveTransform(p.astype(np.float32),rectangle(reference_size).astype(np.float32))
        if not np.isfinite(m).all() or np.linalg.matrix_rank(m)!=3:
            raise ValueError("角点无法形成有效透视变换")
        size = reference_size
        domain = p
        if framing=="margin":
            px,py = round((w-1)*margin_ratio),round((h-1)*margin_ratio)
            size = checked_size((w+2*px,h+2*py))
            m = np.array([[1,0,px],[0,1,py],[0,0,1]],np.float64) @ m
            domain = project_polygon(rectangle(size),np.linalg.inv(m))
            quad(domain,self.output_size,require_inside=False)
        elif framing=="canvas":
            # The entire current input must have a finite projection. Fit its
            # bounding canvas to the requested final width, keeping its aspect.
            domain = rectangle(self.output_size)
            projected = project_polygon(domain,m)
            low = np.floor(projected.min(axis=0))
            high = np.ceil(projected.max(axis=0))
            extent = high-low
            factor = (w-1)/max(extent[0],1)
            factor = min(factor,8190/max(extent[1],1),
                         np.sqrt((MAX_PIXELS-16384)/max(extent[0]*extent[1],1)))
            size = checked_size(tuple(int(np.ceil(v*factor-1e-8))+1 for v in extent))
            m = np.array([[factor,0,-low[0]*factor],[0,factor,-low[1]*factor],[0,0,1]],np.float64) @ m
        self._append("perspective",m.tolist(),size,points=p.tolist(),label=label,details=details or {},
                     reference_size=list(reference_size),framing=framing,margin_ratio=float(margin_ratio),
                     source_domain=domain.tolist())

    def valid_polygon(self):
        polygon = rectangle(self.input_size)
        for op in self.operations:
            if op["kind"]=="crop":
                x,y,w,h = op["rect"]
                domain = rectangle((w,h))+[x,y]
            else:
                domain = np.asarray(op.get("source_domain",op["points"]),np.float64)
            area,clipped = cv2.intersectConvexConvex(polygon.astype(np.float32),np.asarray(domain,np.float32))
            if clipped is None or area<=0:
                return np.empty((0,2),np.float64)
            polygon = project_polygon(clipped.reshape(-1,2),op["matrix"])
            # Homography roundoff at coincident canvas edges can collapse
            # OpenCV's float32 intersection to a line. Snap only subpixel noise.
            for axis,limit in enumerate(np.asarray(op["output_size"])-1):
                polygon[np.abs(polygon[:,axis])<1e-3,axis] = 0
                polygon[np.abs(polygon[:,axis]-limit)<1e-3,axis] = limit
            area,clipped = cv2.intersectConvexConvex(polygon.astype(np.float32),rectangle(op["output_size"]).astype(np.float32))
            if clipped is None or area<=0:
                return np.empty((0,2),np.float64)
            polygon = clipped.reshape(-1,2).astype(np.float64)
        return polygon

    def _append(self, kind, matrix, output_size, **extra):
        if len(self.operations) >= 100:
            raise ValueError("最多保留 100 个操作，请保存后新建项目")
        self.operations.append(dict(kind=kind, matrix=matrix, input_size=list(self.output_size), output_size=list(output_size), **extra))
        self.redo_stack.clear()
        self._mask_cache = None

    def undo(self):
        if self.operations:
            self.redo_stack.append(self.operations.pop())
            self._mask_cache = None

    def redo(self):
        if self.redo_stack:
            self.operations.append(self.redo_stack.pop())
            self._mask_cache = None

    def render(self, frame):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("输入必须是 uint8 BGR 三通道图像")
        if (frame.shape[1], frame.shape[0]) != self.input_size:
            raise ValueError("输入分辨率发生变化，需重新连接或标定；已停止应用旧配置")
        if not self.operations:
            return frame
        if all(op["kind"]=="crop" for op in self.operations):
            # Integer crops are lossless slices, not an opportunity to filter pixels.
            x = sum(op["rect"][0] for op in self.operations)
            y = sum(op["rect"][1] for op in self.operations)
            w,h = self.output_size
            return np.ascontiguousarray(frame[y:y+h,x:x+w])
        image = cv2.warpPerspective(frame,self.matrix,self.output_size,
                                    flags=INTERPOLATIONS[self.interpolation],borderMode=cv2.BORDER_CONSTANT)
        if self.sharpen_amount:
            # Small-radius unsharp mask improves edge contrast; it cannot restore lost letters.
            blurred = cv2.GaussianBlur(image,(0,0),0.7)
            image = cv2.addWeighted(image,1+self.sharpen_amount,blurred,-self.sharpen_amount,0)
        if any(op.get("framing","screen")!="screen" for op in self.operations):
            # Expanded canvases must not resurrect pixels removed by earlier crops.
            if self._mask_cache is None:
                w,h = self.output_size
                self._mask_cache = np.zeros((h,w),np.uint8)
                polygon = self.valid_polygon()
                if len(polygon)>=3:
                    cv2.fillConvexPoly(self._mask_cache,np.rint(polygon*256).astype(np.int32),255,shift=8)
            image = cv2.bitwise_and(image,image,mask=self._mask_cache)
        return image

    def to_dict(self):
        return dict(schema="screenprep.calibration", version=3, rendering=self.render_options,
                    coordinate_system="pixel_centers_xy_top_left", input_size=list(self.input_size),
                    output_size=list(self.output_size), operations=copy.deepcopy(self.operations),
                    source_to_output=self.matrix.tolist(), valid_output_polygon=self.valid_polygon().tolist(), lens_correction=None)

    @classmethod
    def from_dict(cls, doc):
        if doc.get("schema") != "screenprep.calibration" or doc.get("version") not in (1,2,3):
            raise ValueError("不支持的标定文件格式")
        if doc.get("lens_correction") is not None:
            raise ValueError("当前版本暂不支持镜头径向畸变参数")
        options = doc.get("rendering",{})
        if not isinstance(options,dict) or set(options)-{"interpolation","sharpen_amount"}:
            raise ValueError("配置中的采样设置无效")
        if doc["version"]==3 and set(options)!={"interpolation","sharpen_amount"}:
            raise ValueError("配置缺少采样设置")
        result = cls(doc["input_size"],**options)
        ops = doc.get("operations", [])
        if not isinstance(ops, list) or len(ops) > 100:
            raise ValueError("配置操作数量无效")
        for op in ops:
            if tuple(op.get("input_size", ())) != result.output_size:
                raise ValueError("配置中的操作尺寸不连续")
            if op.get("kind") == "crop":
                result.crop(op["rect"])
            elif op.get("kind") == "perspective":
                if doc["version"]==1 and op.get("framing","screen")!="screen":
                    raise ValueError("扩展画面范围需要第 2 版配置格式")
                result.perspective(op["points"],op.get("reference_size",op["output_size"]),
                                   op.get("label","四点校正"),op.get("details",{}),
                                   framing=op.get("framing","screen"),margin_ratio=op.get("margin_ratio",0.0))
            else:
                raise ValueError("配置中含未知操作")
            if tuple(op.get("output_size", ())) != result.output_size:
                raise ValueError("配置输出尺寸不一致")
        if tuple(doc.get("output_size", ())) != result.output_size:
            raise ValueError("配置最终尺寸不一致")
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 2_000_000:
            raise ValueError("配置文件过大")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
