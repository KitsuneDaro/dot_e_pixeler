import ImageInput from './ImageInput';
import ConvertingButton from './ConvertingButton';
import Explain from './Explain';
import Parameter from './Parameter';
import Unit from './Unit';

import noImage from '../assets/noimage.png';
import SaveButton from './SaveButton';

function DrawImg({ param }: { param: Parameter }) {
    return (
        <div className='flex row'>
            <Unit className='flex column'>
                <Explain>画像読み込み</Explain>
                <ImageInput drawingQuery="#original_image_img" param={param}></ImageInput>
                <img id="original_image_img" src={noImage}></img>
            </Unit>

            <Unit className='flex column'>
                <Explain>高速な変換</Explain>
                <div className='flex row'>
                    <ConvertingButton
                        param={param}
                        mode="kmeans"
                        width={param.shrinkedWidth}
                        height={param.shrinkedHeight}
                        expandingScale={param.expandingScale}
                        convertingQuery="#original_image_img"
                        convertedQuery="#kmeans_converted_image_img"
                    ></ConvertingButton>
                    <SaveButton param={param} imgQuery="#kmeans_converted_image_img"></SaveButton>
                </div>
                <img id="kmeans_converted_image_img"></img>
            </Unit>

            <Unit className='flex column'>
                <Explain>色を偏らせる変換</Explain>
                <div className='flex row'>
                    <ConvertingButton
                        param={param}
                        mode="mix"
                        width={param.shrinkedWidth}
                        height={param.shrinkedHeight}
                        expandingScale={param.expandingScale}
                        convertingQuery="#original_image_img"
                        convertedQuery="#mix_converted_image_img"
                    ></ConvertingButton>
                    <SaveButton param={param} imgQuery="#mix_converted_image_img"></SaveButton>
                </div>
                <img id="mix_converted_image_img"></img>
            </Unit>
        </div>
    );
}

export default DrawImg