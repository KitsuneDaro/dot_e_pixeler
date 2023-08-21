import Parameter from './Parameter';
import Explain from './Explain';
import Unit from './Unit';
import IntInput from './IntInput';

function SizeSetting({ param }: { param: Parameter }) {
    const widthEvent = () => { //(event: React.ChangeEvent<InputEvent>) => {
        const input: any = document.querySelector('#shrinked_width');
        const shrinkedWidth = parseInt(input.value);

        param.setShrinkedWidth(shrinkedWidth);

        param.setShrinkingScale(param.width / shrinkedWidth);
        param.setShrinkedHeight(Math.ceil(param.height / (param.width / shrinkedWidth)));
    };

    const heightEvent = () => { //(event: React.ChangeEvent<InputEvent>) => {
        const input: any = document.querySelector('#shrinked_height');
        const shrinkedHeight = parseInt(input.value);

        param.setShrinkedHeight(shrinkedHeight);
        
        param.setShrinkingScale(param.height / shrinkedHeight);
        param.setShrinkedWidth(Math.ceil(param.width / (param.height / shrinkedHeight)));
    };

    return (
        <>
            <Unit className='flex row'>
                <div className='flex column'>
                    <Explain>縮小後横幅(pixel)</Explain>
                    <IntInput id="shrinked_width" value={param.shrinkedWidth} onChange={widthEvent}></IntInput>
                </div>
                <div className='flex column'>
                    <Explain>縮小後縦幅(pixel)</Explain>
                    <IntInput id="shrinked_height" value={param.shrinkedHeight} onChange={heightEvent}></IntInput>
                </div>
            </Unit>
        </>
    );
}

export default SizeSetting