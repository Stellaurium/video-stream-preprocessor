import copy
import cv2
import numpy as np
import pytest
from screenprep.geometry import Pipeline


def test_lanczos_preserves_bandlimited_detail_better_than_bilinear():
    # Known continuous signal provides a reference independent of either sampler.
    h,w=160,240
    y,x=np.mgrid[:h,:w].astype(np.float64)
    def signal(x,y):
        return 128+60*np.sin(2*np.pi*.19*x)+40*np.cos(2*np.pi*.14*y)
    image=np.repeat(np.rint(signal(x,y)).clip(0,255).astype(np.uint8)[:,:,None],3,axis=2)
    pipeline=Pipeline((w,h),sharpen_amount=0)
    pipeline.perspective([[7.3,5.2],[231.6,10.4],[229.3,152.8],[10.2,150.7]],(320,210))
    yo,xo=np.mgrid[:210,:320]
    coords=cv2.perspectiveTransform(np.dstack((xo,yo)).astype(np.float64).reshape(-1,1,2),np.linalg.inv(pipeline.matrix)).reshape(210,320,2)
    reference=signal(coords[:,:,0],coords[:,:,1])[20:-20,20:-20]
    errors={}
    for method in ['linear','cubic','lanczos4']:
        pipeline.set_render_options(method,0)
        actual=pipeline.render(image)[20:-20,20:-20,0]
        errors[method]=np.mean(np.abs(actual-reference))
    assert errors['lanczos4'] < errors['linear']*.4,errors
    assert errors['cubic'] < errors['linear']*.5,errors


def test_quality_roundtrip_and_clone_preserve_pixels_and_geometry(tmp_path):
    image=np.random.default_rng(7).integers(0,256,(180,320,3),dtype=np.uint8)
    pipeline=Pipeline((320,180),'cubic',.35)
    pipeline.perspective([[15,10],[300,8],[305,165],[10,170]],(400,225))
    pipeline.crop((10,10,360,200))
    matrix=pipeline.matrix.copy()
    expected=pipeline.render(image)
    path=tmp_path/'quality.json';pipeline.save(path)
    for restored in [pipeline.clone(),Pipeline.load(path)]:
        assert restored.render_options==dict(interpolation='cubic',sharpen_amount=.35)
        np.testing.assert_allclose(restored.matrix,matrix)
        np.testing.assert_array_equal(restored.render(image),expected)
    pipeline.set_render_options('linear',0)
    np.testing.assert_allclose(pipeline.matrix,matrix)
    assert not np.array_equal(pipeline.render(image),expected)
    for version in [1,2]:
        old=copy.deepcopy(pipeline.to_dict());old['version']=version;old.pop('rendering')
        restored=Pipeline.from_dict(old)
        assert restored.interpolation=='lanczos4' and restored.sharpen_amount==.2
        np.testing.assert_allclose(restored.matrix,matrix)


def test_unmodified_and_crop_only_frames_are_lossless_even_with_sharpening():
    image=np.random.default_rng(5).integers(0,256,(180,320,3),dtype=np.uint8)
    pipeline=Pipeline((320,180),sharpen_amount=.6)
    assert pipeline.render(image) is image
    pipeline.crop((10,20,290,140));pipeline.crop((5,7,200,100))
    np.testing.assert_array_equal(pipeline.render(image),image[27:127,15:215])


@pytest.mark.parametrize('method,amount',[('invalid',0),('linear',-1),('cubic',float('nan')),('lanczos4',.61),('linear',True)])
def test_invalid_quality_settings_do_not_mutate_pipeline(method,amount):
    pipeline=Pipeline((100,100))
    before=pipeline.render_options
    with pytest.raises(ValueError):pipeline.set_render_options(method,amount)
    assert pipeline.render_options==before


def test_fractional_quad_has_full_valid_canvas_after_correction():
    # Real four-point coordinates can project a few microunits past the canvas.
    pipeline=Pipeline((1920,1080))
    pipeline.perspective([[422.70450751252076,144.24040066777962],
                          [1443.2053422370618,97.36227045075125],
                          [1540.5676126878132,818.5642737896494],
                          [541.702838063439,1024.1068447412354]],(1920,1080))
    polygon=pipeline.valid_polygon()
    assert len(polygon)>=4
    assert cv2.contourArea(polygon.astype(np.float32))>2_000_000
