import Parameter from './Parameter';
import Explain from './Explain';
import Unit from './Unit';
import IntInput from './IntInput';
import FloatInput from './FloatInput';

function ScaleSetting({ param }: { param: Parameter }) {
    const shrinkingScaleEvent = () => {//(event: React.ChangeEvent<InputEvent>) => {
        const input: any = document.querySelector('#shrinking_scale');
        param.setShrinkingScale(parseInt(input.value));

        param.setShrinkedWidth(Math.ceil(param.width / parseInt(input.value)));
        param.setShrinkedHeight(Math.ceil(param.height / parseInt(input.value)));
    };

    const expandingScaleEvent = () => {//(event: React.ChangeEvent<InputEvent>) => {
        const input: any = document.querySelector('#expanding_scale');
        param.setExpandingScale(parseInt(input.value));
    };

    return (
        <>
            <Unit className='flex row'>
                <div className='flex column'>
                    <Explain>縮小率(倍)</Explain>
                    <FloatInput id="shrinking_scale" value={param.shrinkingScale} onChange={shrinkingScaleEvent}></FloatInput>
                </div>
                <div className='flex column'>
                    <Explain>拡大率(倍)</Explain>
                    <IntInput id="expanding_scale" value={param.expandingScale} onChange={expandingScaleEvent}></IntInput>
                </div>
            </Unit>
        </>
    );
}

export default ScaleSetting