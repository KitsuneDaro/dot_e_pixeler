import { GMM } from "./GMMbyWebGL/src/GMM";
import * as GPGPU from "./GMMbyWebGL/src/gpgpu";
import { kmeansInc } from './GMMbyWebGL/src/KmeansInc';

export class ClusteringImageRGB {
    static ChangeImgByCanvas(canvas: HTMLCanvasElement, img: HTMLImageElement): undefined {
        const dataURI = canvas.toDataURL();

        img.src = dataURI;
    }

    static ChangeCanvasByImg(img: HTMLImageElement, canvas: HTMLCanvasElement, width: number, height: number): undefined {
        const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
        canvas.width = width;
        canvas.height = height;
        ctx.imageSmoothingEnabled = false;

        ctx.drawImage(img, 0, 0, width, height);
    }

    static DrawCanvasRGBAData(width: number, height: number, rgba_data: Uint8ClampedArray, canvas: HTMLCanvasElement): undefined {
        const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
        const imageData = new ImageData(rgba_data, width, height);
        
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        ctx.putImageData(imageData, 0, 0);
    }

    static ScaleCanvas(canvas: HTMLCanvasElement, scale: number, handler: Function = () => {}): undefined {
        const img = new Image();

        img.onload = () => {
            canvas.width *= scale;
            canvas.height *= scale;

            const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
            
            ctx.imageSmoothingEnabled = false;

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            handler();
        };

        img.src = canvas.toDataURL();
    }

    // muから作るやつ
    static GetClusteringRGBDataByMu(rgba_data: Uint8ClampedArray, cluster_mu: Float32Array): Uint8ClampedArray {
        const gpgpu = GPGPU.CreateGPGPU();

        const dist_n = Math.round(cluster_mu.length / 3);
        const data_n = Math.round(rgba_data.length / 4);

        const parsed_data_n = gpgpu.getMaxTextureSize();
        const parsed_data_n_n = Math.ceil(data_n / parsed_data_n);

        const clustering_rgb_data_shader = `
            uniform vec3 cluster_mu[${dist_n}];

            in vec4 rgba_data;
            out vec4 new_rgba_data;

            void main() {
                if (rgba_data[0] == -1.0) {
                    return;
                }

                vec3 rgb_data = vec3(rgba_data[0], rgba_data[1], rgba_data[2]);
                vec3 rgb_delta = (rgb_data - cluster_mu[0]);

                float min_var = dot(rgb_delta, rgb_delta);
                int min_index = 0;

                for (int k = 1; k < ${dist_n}; k++) {
                    vec3 rgb_delta = (rgb_data - cluster_mu[k]);

                    float var = dot(rgb_delta, rgb_delta);

                    if (var < min_var) {
                        min_var = var;
                        min_index = k;
                    }
                }

                rgb_data = cluster_mu[min_index];

                new_rgba_data = vec4(rgb_data[0], rgb_data[1], rgb_data[2], rgba_data[3]);
            }
        `;

        const parsed_rgba_data = new Float32Array(parsed_data_n * 4);
        const new_rgba_data = new Float32Array(rgba_data.length);

        const clustering_rgb_data_param = {
            id: 'clustering_rgb_data_shader',
            vertexShader: clustering_rgb_data_shader,
            args: {
                'cluster_mu': cluster_mu,
                'rgba_data': parsed_rgba_data,
                'new_rgba_data': parsed_rgba_data
            }
        }

        for (let i = 0; i < parsed_data_n_n; i++) {
            if (i == parsed_data_n_n - 1) {
                parsed_rgba_data.fill(-1.0);
            }

            parsed_rgba_data.set(rgba_data.slice(i * parsed_data_n * 4, (i + 1) * parsed_data_n * 4), 0);

            gpgpu.compute(clustering_rgb_data_param);

            if (i < parsed_data_n_n - 1) {
                new_rgba_data.set(parsed_rgba_data, i * parsed_data_n * 4);
            } else {
                new_rgba_data.set(parsed_rgba_data.slice(0, data_n * 4 - i * parsed_data_n * 4), i * parsed_data_n * 4);
            }
        }

        //const test = rgba_data.slice(rgba_data.length / 2 + 256 * 16, rgba_data.length * 3 / 4);

        gpgpu.clear(clustering_rgb_data_param.id);

        return new Uint8ClampedArray(new_rgba_data);
    }

    // GMM + kmeans Clustering
    static GetMuByGMMAndKmeansClusteringRGBData(gmm_dist_n: number, kmeans_dist_n: number, rgb_data: Float32Array): Float32Array {
        // GMMでクリスタリング
        const gmm_cluster_dict = this.GetGMMClusteringRGBData(gmm_dist_n, rgb_data);
        const dist_n = gmm_dist_n + kmeans_dist_n;
//        const data_n = Math.round(rgb_data.length / 3);

        const sigma_abs_mean_sqrt_shader = `
            uniform mat3 sigma[${dist_n}];

            in float zero;
            out float sams;

            bool isNaN(float val) {
                return (val < 0.0 || val == 0.0 || 0.0 < val ) ? false : true;
            }

            void main() {
                float value = sqrt(determinant(sigma[gl_VertexID]));

                if (isNaN(value)) {
                    sams = zero;
                } else {
                    sams = value;
                }
            }
        `;

        const gpgpu = GPGPU.CreateGPGPU();
        const dist_n_zero = new Float32Array(gmm_dist_n);
        const sams = new Float32Array(gmm_dist_n);

        const sigma_abs_mean_sqrt_param = {
            id: 'sigma_abs_mean_sqrt_shader',
            vertexShader: sigma_abs_mean_sqrt_shader,
            args: {
                'sigma': gmm_cluster_dict.gmm.sigma,
                'zero': dist_n_zero,
                'sams': sams
            }
        };

        gpgpu.compute(sigma_abs_mean_sqrt_param);
        gpgpu.clear(sigma_abs_mean_sqrt_param.id);

        const sams_sum = sams.reduce((x, y) => x + y);
        
        const kmeans_dist_ns = Uint32Array.from(sams, (value: number) => {
            return Math.ceil((value / sams_sum) * (kmeans_dist_n - gmm_dist_n)) + 1;
        });
        
        //const kmeans_dist_ns = Uint32Array.from(gmm_cluster_dict.divided_data_n, (value: number) => { return Math.ceil(value / data_n) });
        const kmeans_dist_ns_sum = kmeans_dist_ns.reduce((x, y) => x + y);

        const cluster_mu = new Float32Array(dist_n * 3);

        for (let i = 0; i < kmeans_dist_ns_sum - kmeans_dist_n; i++) {
            const index = sams.indexOf(Math.min(...sams));

            kmeans_dist_ns[index] -= 1;
            sams[index] = Infinity;
        }


        var dist_k_sum = 0;
        var try_n = 0;

        for (let i = 0; i < gmm_dist_n; i++) {
            const dist_k = kmeans_dist_ns[i] + 1;
            const data_n = gmm_cluster_dict.divided_data_n[i];

            const divided_rgb_data = gmm_cluster_dict.divided_rgb_data[i];

            if (dist_k > 1) {
                try {
                    const mu_x_cluster = kmeansInc(dist_k, data_n, divided_rgb_data);
                    const mu = mu_x_cluster[0];

                    cluster_mu.set(mu, dist_k_sum * 3);

                    dist_k_sum += dist_k;
                } catch {
                    const rand = Math.random();
                    const temp_k = Math.floor(rand * (gmm_dist_n - 1));
                    const k = (temp_k < i) ? temp_k : temp_k + 1;
                    
                    kmeans_dist_ns[k] += kmeans_dist_ns[i]
                    kmeans_dist_ns[i] = 0;

                    dist_k_sum = 0;
                    i = -1;

                    try_n += 1;
                }
            } else {
                const mu = gmm_cluster_dict.gmm.mu.slice(i * 3, (i + 1) * 3);
                
                cluster_mu.set(mu, dist_k_sum * 3);

                dist_k_sum += dist_k;
            }

            if (try_n > gmm_dist_n * 20) {
                throw new Error('kmeansの仕分け色が多すぎる可能性があります。');
            }
        }

        return cluster_mu;
    }

    // GMM Clustering
    static GetGMMClusteringRGBData(dist_n: number, rgb_data: Float32Array): {
        gmm: GMM, cluster: Uint32Array, divided_rgb_data: Float32Array[], divided_data_n: Uint32Array
    } {
        const data_n = Math.floor(rgb_data.length / 3);

        // GMMのフィッティング
        const gmm_gamma_dict = GMM.CreateGMM(dist_n, data_n, rgb_data, 0.01, 100);

        // 事後確率からクラスタリング
        const gmm_post_prob = GMM.PostProbByGamma(dist_n, data_n, gmm_gamma_dict.gamma);
        const gmm_cluster = GMM.ClusteringByPostProb(dist_n, data_n, gmm_post_prob);

        // クラスタリングに基づいてデータを分割
        const divided_dict = GMM.DivideDataByCluster(dist_n, data_n, rgb_data, gmm_cluster);

        return {
            gmm: gmm_gamma_dict.gmm,
            cluster: gmm_cluster,
            divided_rgb_data: divided_dict.divided_x,
            divided_data_n: divided_dict.divided_data_n
        };
    }

    // 透明部分を抜かしたRGBデータと透明度のみのデータを取得
    static GetClipedRGBDataAndADataByRGBAData(rgba_data: Uint8ClampedArray): {
        cliped_rgb_data: Float32Array, a_data: Uint8ClampedArray
    } {
        const cliped_rgb_data = new Array(0);
        const a_data = new Uint8ClampedArray(rgba_data.length / 4);
        
        for (let i = 0; i < rgba_data.length; i += 4) {
            const a = rgba_data[i + 3];

            if (a > 0) {
                cliped_rgb_data.push(...rgba_data.slice(i, i + 3)); // RGBのみ
            }

            a_data[i / 4] = a;
        }

        return {cliped_rgb_data: new Float32Array(cliped_rgb_data), a_data: a_data};
    }

    // Canvas要素からRGBAデータを取得
    static GetImageDataByCanvas(canvas: HTMLCanvasElement): ImageData {
        const ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
        const image_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        return image_data;
    }

    // 画像縮小
    static GetShrinkedImageByCanvas(canvas: HTMLCanvasElement, width: number, height: number): ImageData {
        const tempCanvas: HTMLCanvasElement = document.createElement('canvas');

        tempCanvas.width = width;
        tempCanvas.height = height;

        const tempCtx = <CanvasRenderingContext2D>tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, width, height);
        
        return tempCtx.getImageData(0, 0, width, height);
    }
}