import { useState } from 'react';
import { CreateGPGPU } from '../ClusteringImageRGB/GMMbyWebGL/src/gpgpu';

class Parameter {
    shrinkingScale: number;
    setShrinkingScale: Function;

    expandingScale: number;
    setExpandingScale: Function;

    width: number;
    setWidth: Function;

    height: number;
    setHeight: Function;

    shrinkedWidth: number;
    setShrinkedWidth: Function;

    shrinkedHeight: number;
    setShrinkedHeight: Function;

    colorNum: number;
    setColorNum: Function;

    fileName: string;
    setFileName: Function;

    maxTextureSize: number;

    loadingQuery: string;

    loadingHidden: boolean;
    setLoadingHidden: Function;
    
    constructor() {
        const [shrinkingScale, setShrinkingScale] = useState(2);

        this.shrinkingScale = shrinkingScale;
        this.setShrinkingScale = setShrinkingScale;

        const [expandingScale, setExpandingScale] = useState(8);

        this.expandingScale = expandingScale;
        this.setExpandingScale = setExpandingScale;

        const [width, setWidth] = useState(NaN);

        this.width = width;
        this.setWidth = setWidth;

        const [height, setHeight] = useState(NaN);

        this.height = height;
        this.setHeight = setHeight;

        const [shrinkedWidth, setShrinkedWidth] = useState(NaN);

        this.shrinkedWidth = shrinkedWidth;
        this.setShrinkedWidth = setShrinkedWidth;

        const [shrinkedHeight, setShrinkedHeight] = useState(NaN);

        this.shrinkedHeight = shrinkedHeight;
        this.setShrinkedHeight = setShrinkedHeight;

        const [colorNum, setColorNum] = useState(8);

        this.colorNum = colorNum;
        this.setColorNum = setColorNum;

        const [fileName, setFileName] = useState('');

        this.fileName = fileName;
        this.setFileName = setFileName;

        const gpgpu = CreateGPGPU();

        this.maxTextureSize = gpgpu.getMaxTextureSize();

        this.loadingQuery = '#loading';

        const [loadingHidden, setLoadingHidden] = useState(true);

        this.loadingHidden = loadingHidden;
        this.setLoadingHidden = setLoadingHidden;
    }
}

export default Parameter