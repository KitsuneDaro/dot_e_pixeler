import Parameter from './Parameter';
import Explain from './Explain';
import Unit from './Unit';
import IntInput from './IntInput';

function ColorSetting({ param }: { param: Parameter }) {
    const colorNumEvent = () => { //(event: React.ChangeEvent<InputEvent>) => {
        const input: any = document.querySelector('#color_num');

        param.setColorNum(parseInt(input.value));
    };

    return (
        <>
            <Unit className='flex row'>
                <div className='flex column'>
                    <Explain>色数(色)</Explain>
                    <IntInput id="color_num" value={param.colorNum} onChange={colorNumEvent} max={32} min={4}></IntInput>
                </div>
            </Unit>
        </>
    );
}

export default ColorSetting