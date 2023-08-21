import React, { useRef } from 'react';
import Button from './Button';
import Parameter from './Parameter';

function ImageInput({ drawingQuery, param }: { drawingQuery: string, param: Parameter }) {
    const inputRef = useRef<HTMLInputElement>(null);

    const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files instanceof FileList) {
            if (e.target.files[0]) {
                const img: any = document.querySelector<HTMLCanvasElement>(drawingQuery);

                // ファイル読み込み完了
                const reader = new FileReader();
        
                reader.onload = () => {
                    if (typeof (reader.result) == 'string') {
                        img.src = reader.result;
        
                        img.onload = () => {
                            param.setWidth(img.naturalWidth);
                            param.setHeight(img.naturalHeight);
        
                            param.setShrinkedWidth(Math.ceil(img.naturalWidth / param.shrinkingScale));
                            param.setShrinkedHeight(Math.ceil(img.naturalHeight / param.shrinkingScale));
                        }
                    }
                };
                
                reader.readAsDataURL(e.target.files[0]);
                param.setFileName(e.target.files[0].name);
            }
        }
    };

    const onFileUpload = () => {
        if (inputRef.current) {
            inputRef.current.click();
        }
    }

    return <>
        <Button innerText="読込" onClick={onFileUpload}></Button>
        <input
            type="file"
            accept="image/*"
            id="image_input"
            onChange={onFileInputChange}
            hidden
            ref={inputRef}
        ></input>
    </>;
};

export default ImageInput