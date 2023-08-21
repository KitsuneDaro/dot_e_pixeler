import Button from "./Button";
import Parameter from "./Parameter";

function SaveButton({ param, imgQuery }: { param: Parameter, imgQuery: string }) {
    const img: any = document.querySelector(imgQuery);
    
    const saveEvent = () => {
        if (!param.fileName || img.naturalWidth == 0) {
            alert('画像が変換されていません！');
            return;
        }

        const aElement = document.createElement('a');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx?.drawImage(img, 0, 0);

        aElement.href = canvas.toDataURL('image/png');

        const splitFileName = param.fileName.split('.');
        
        aElement.download = splitFileName.reduce((pre, cur, index) => {
            if (index > 1) {
                return pre + '.' + cur;
            } else {
                return pre + '_copy.' + cur;
            }
        });
        aElement.click();
    }


    return (
        <Button innerText="保存" onClick={saveEvent}></Button>
    );
}

export default SaveButton