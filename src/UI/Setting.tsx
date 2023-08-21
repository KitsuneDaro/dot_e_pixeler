import ColorSetting from './ColorSetting';
import Explain from './Explain';
import Parameter from './Parameter';
import ScaleSetting from './ScaleSetting';
import SizeSetting from './SizeSetting';
import Unit from './Unit';

function Setting({ param }: { param: Parameter }) {
    return (
        <>
            <div className='flex row'>
                <ScaleSetting param={param}></ScaleSetting>
                <SizeSetting param={param}></SizeSetting>
                <ColorSetting param={param}></ColorSetting>
                <Unit className='flex column'>
                    <Explain>
                        画像サイズ(pixel^2)
                    </Explain>
                    <div className='flex row'>
                            <Explain>
                                {'現在の縮小後サイズ:' + (param.shrinkedWidth * param.shrinkedHeight).toLocaleString()}
                            </Explain>
                            <Explain>
                                {'変換可能な最大サイズ:' + (Math.floor(param.maxTextureSize / Math.ceil(param.colorNum / 3)) * param.maxTextureSize).toLocaleString()}
                            </Explain>
                    </div>
                </Unit>
            </div>
        </>
    );
}

export default Setting