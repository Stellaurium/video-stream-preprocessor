import cv2
import numpy as np
import pytest
from screenprep.geometry import Pipeline,rectangle
from screenprep.calibration import generate_board,detect_board


def small_window_scene():
    plane=np.full((720,1280,3),190,np.uint8)
    plane[180:540,400:1040]=cv2.resize(generate_board(),(640,360))
    cv2.rectangle(plane,(100,100),(190,155),(30,210,30),-1)
    cv2.rectangle(plane,(1090,590),(1190,660),(220,50,20),-1)
    h=cv2.getPerspectiveTransform(rectangle((1280,720)).astype(np.float32),np.float32([[80,70],[1170,30],[1230,730],[40,770]]))
    camera=cv2.warpPerspective(plane,h,(1280,800))
    return camera,h


def test_windowed_standard_image_preserves_surrounding_notes_and_roundtrip(tmp_path):
    camera,h=small_window_scene()
    points,size,details=detect_board(camera,1280)
    p=Pipeline((1280,800))
    p.perspective(points,size,'自动',details,framing='canvas')
    assert p.output_size[0]==1280
    mapped=cv2.perspectiveTransform(rectangle((1280,800)).reshape(-1,1,2),p.matrix).reshape(-1,2)
    assert mapped.min()>=-.01
    assert mapped[:,0].max()<=p.output_size[0]-1+.01
    assert mapped[:,1].max()<=p.output_size[1]-1+.01
    image=p.render(camera)
    for point,expected in [((145,125),(30,210,30)),((1140,625),(220,50,20))]:
        mapped_note=cv2.perspectiveTransform(np.float64(point).reshape(1,1,2),p.matrix@h).reshape(2)
        x,y=np.rint(mapped_note).astype(int)
        np.testing.assert_allclose(image[y,x],expected,atol=5)
    path=tmp_path/'留边配置.json'
    p.save(path)
    reloaded=Pipeline.load(path)
    np.testing.assert_array_equal(reloaded.render(camera),image)
    assert reloaded.to_dict()['version']==3
    before=p.matrix.copy()
    p.crop((10,10,p.output_size[0]-20,p.output_size[1]-20))
    p.undo()
    np.testing.assert_allclose(p.matrix,before)
    p.redo()
    assert p.output_size[0]==1260


def test_margin_is_per_side_and_samples_real_surrounding_pixels():
    image=np.full((360,640,3),(25,155,80),np.uint8)
    image[140:230,240:400]=(200,40,60)
    p=Pipeline((640,360))
    p.perspective([[240,140],[399,140],[399,229],[240,229]],(160,90),framing='margin',margin_ratio=.2)
    assert p.output_size==(224,126)
    result=p.render(image)
    np.testing.assert_array_equal(result[1,1],[25,155,80])
    np.testing.assert_array_equal(result[63,112],[200,40,60])
    p.undo()
    assert p.output_size==(640,360)
    p.redo()
    np.testing.assert_array_equal(p.render(image),result)


def test_expansion_never_restores_pixels_discarded_by_previous_crop(tmp_path):
    image=np.full((600,1000,3),(0,0,255),np.uint8)
    image[100:500,200:800]=(20,180,30)
    p=Pipeline((1000,600))
    p.crop((200,100,600,400))
    p.perspective([[10,10],[589,10],[589,389],[10,389]],(580,380),framing='margin',margin_ratio=.2)
    result=p.render(image)
    np.testing.assert_array_equal(result[result.shape[0]//2,1],[0,0,0])
    np.testing.assert_array_equal(result[result.shape[0]//2,result.shape[1]//2],[20,180,30])
    # No red pixels from outside the earlier crop can leak into the extended canvas.
    assert not ((result[:,:,2]>200)&(result[:,:,1]<20)).any()
    width,height=p.output_size
    p.perspective([[2,2],[width-3,2],[width-3,height-3],[2,height-3]],(640,400),framing='canvas')
    result=p.render(image)
    assert not ((result[:,:,2]>200)&(result[:,:,1]<20)).any()
    path=tmp_path/'combined.json'
    p.save(path)
    np.testing.assert_array_equal(Pipeline.load(path).render(image),result)


def test_unbounded_full_canvas_rejected_without_changing_pipeline():
    points=[[450,300],[550,300],[650,500],[350,500]]
    p=Pipeline((1000,800))
    with pytest.raises(ValueError,match='无穷远'):
        p.perspective(points,(640,360),framing='canvas')
    assert not p.operations
    p.perspective(points,(640,360),framing='margin',margin_ratio=.1)
    assert p.output_size[0]>640


def test_legacy_screen_configs_still_load():
    p=Pipeline((640,360))
    p.perspective([[30,20],[600,25],[590,330],[35,335]],(640,360))
    doc=p.to_dict()
    doc['version']=1
    for name in ['reference_size','framing','margin_ratio','source_domain']:
        doc['operations'][0].pop(name)
    restored=Pipeline.from_dict(doc)
    np.testing.assert_allclose(restored.matrix,p.matrix)
    assert restored.output_size==(640,360)
