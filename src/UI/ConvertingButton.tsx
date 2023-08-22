import Button from './Button';
import { ClusteringImageRGB } from '../ClusteringImageRGB/ClusteringImageRGB';
// import { GMM } from '../ClusteringImageRGB/GMMbyWebGL/src/GMM';
import { kmeansInc } from '../ClusteringImageRGB/GMMbyWebGL/src/KmeansInc';
import Parameter from './Parameter';

function ConvertingButton(
    { param, convertingQuery, convertedQuery, mode, width, height, expandingScale }:
        { param: Parameter, convertingQuery: string, convertedQuery: string, mode: 'mix' | 'kmeans', width: number, height: number, expandingScale: number }
) {
    const onClick = () => {
        if (!param.colorNum) {
            alert('色数が指定されていません！');
            return;
        }

        if (!param.fileName) {
            alert('画像が選択されていません！');
            return;
        }

        if (param.shrinkedWidth * param.shrinkedHeight > Math.floor(param.maxTextureSize / Math.ceil(param.colorNum / 3)) * param.maxTextureSize) {
            alert('画像サイズ、あるいは色数が多すぎます！画像サイズを最大サイズよりも小さくしてください！！');
            return;
        }

        const dist_n = Math.min(Math.max(param.colorNum, 4), 32);

        const convertingImg: any = document.querySelector(convertingQuery);
        const convertingCanvas = document.createElement('canvas');

        const gmm_dist_n = Math.ceil(dist_n / 3);
        const kmeans_dist_n = dist_n - gmm_dist_n;

        ClusteringImageRGB.ChangeCanvasByImg(convertingImg, convertingCanvas, width, height);

        // 1. 画像縮小
        const rgbaImageData: ImageData = ClusteringImageRGB.GetShrinkedImageByCanvas(convertingCanvas, width, height);
        const rgbAndADataDict = ClusteringImageRGB.GetClipedRGBDataAndADataByRGBAData(rgbaImageData.data);

        // 2. クラスタリング GetMuByGMMAndKmeansClusteringRGBData
        var clusteredMu;

        try {
            if (mode == 'kmeans') {
                const kmeansArray = kmeansInc(dist_n, Math.round(rgbAndADataDict.cliped_rgb_data.length / 3), rgbAndADataDict.cliped_rgb_data);
                clusteredMu = kmeansArray[0];
            } else {
                clusteredMu = ClusteringImageRGB.GetMuByGMMAndKmeansClusteringRGBData(gmm_dist_n, kmeans_dist_n, rgbAndADataDict.cliped_rgb_data);
            }

            // 3. 画像に適用 GetClusteringRGBDataByMu
            const convertedRgbaData = ClusteringImageRGB.GetClusteringRGBDataByMu(rgbaImageData.data, clusteredMu);

            const convertedCanvas = document.createElement('canvas');
            ClusteringImageRGB.DrawCanvasRGBAData(width, height, convertedRgbaData, convertedCanvas);
            ClusteringImageRGB.ScaleCanvas(convertedCanvas, expandingScale, () => {
                const convertedImg: any = document.querySelector(convertedQuery);
                ClusteringImageRGB.ChangeImgByCanvas(convertedCanvas, convertedImg);

                convertedCanvas.onresize = null;
            });
        } catch {
            alert('画像の含む色に対して変換後の色数が多すぎる可能性があります！また、ブラウザがChrome、Edge以外の場合には正常に動かない可能性があります！');
        }
    }

    return <Button onClick={async () => {
        param.setLoadingHidden(false);

        setTimeout(() => {
            onClick();
            param.setLoadingHidden(true);
        }, 0);
    }} innerText="変換"></Button>
}

export default ConvertingButton