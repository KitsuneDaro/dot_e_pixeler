import * as GPGPU from './gpgpu';
import { kmeansInc } from './KmeansInc';

export class GMM {
    /* GMM class (3D限定)*/

    dist_n: number;
    mu: Float32Array;
    pi: Float32Array;
    sigma: Float32Array;

    constructor(dist_n: number, mu: Float32Array, pi: Float32Array, sigma: Float32Array) {
        this.dist_n = dist_n;

        this.mu = mu;
        this.pi = pi;
        this.sigma = sigma;
    }

    /* constructer input check */

    static CheckMu(dist_n: number, mu: Float32Array): boolean {
        return dist_n == mu.length;
    }

    static CheckPi(dist_n: number, pi: Float32Array): boolean {
        return dist_n == pi.length;
    }

    static CheckSigma(dist_n: number, sigma: Float32Array): boolean {
        return dist_n * dist_n == sigma.length;
    }

    // GMMを作るやつ
    static CreateGMM(
        dist_n: number, data_n: number, x: Float32Array, regularation_value: number = 0.0, max_iteration_time: number = 100, log_p_tolerance_value: number = 1e-6
    ): { gmm: GMM, gamma: Float32Array, gamma_sum: Float32Array } {
        const gpgpu = GPGPU.CreateGPGPU();

        // Initialize
        const x_mu_std = this.EvalXMuStd(data_n, x);
        const x_std = x_mu_std[1];

        const init_mu = this.InitMu(dist_n, data_n, x);
        const init_pi = this.InitPi(dist_n);
        const init_sigma = this.InitSigma(dist_n, x_std);

        // Variables
        const max_texture_size = gpgpu.getMaxTextureSize();

        const vec3_texture_m = Math.floor(max_texture_size / 3);
        const vec3_texture_w = 3 * vec3_texture_m;
        const vec3_texture_h = Math.ceil(data_n / vec3_texture_m);
        const vec3_texture_len = vec3_texture_w * vec3_texture_h;

        const dist_n_texture_m = Math.floor(max_texture_size / dist_n);
        const dist_n_texture_w = dist_n * dist_n_texture_m;
        const dist_n_texture_h = Math.ceil(data_n / dist_n_texture_m);
        const dist_n_texture_len = dist_n_texture_w * dist_n_texture_h;

        // 例外処理
        if (dist_n_texture_h > max_texture_size) {
            throw new Error('データサイズが大きすぎます');
        }

        // Shaders
        // const inv_sigma_shader = 

        const norm_x_shader = `
            uniform sampler2D x; // parsed_data_n * 3

            uniform vec3 mu[${dist_n}];
            uniform mat3 sigma[${dist_n}];

            in float zero;
            out float norm_x;// padding必要あり

            bool isNaN(float val) {
                return (val < 0.0 || val == 0.0 || 0.0 < val ) ? false : true;
            }
    
            vec3 getSampler2DVec3(sampler2D data, int n) {
                int x = (n % ${vec3_texture_m}) * 3;
                int y = n / ${vec3_texture_m};

                return vec3(texelFetch(data, ivec2(x, y), 0).r, texelFetch(data, ivec2(x + 1, y), 0).r, texelFetch(data, ivec2(x + 2, y), 0).r);
            }

            float normdist(vec3 x, vec3 mu, mat3 sigma) {
                vec3 nx = x - mu;
                mat3 invsigma = inverse(sigma); // ここ分けてもよさそう
                
                float s2d = nx[0] * (invsigma[0][0] * nx[0] + invsigma[0][1] * nx[1] + invsigma[0][2] * nx[2]) + nx[1] * (invsigma[1][0] * nx[0] + invsigma[1][1] * nx[1] + invsigma[1][2] * nx[2]) + nx[2] * (invsigma[2][0] * nx[0] + invsigma[2][1] * nx[1] + invsigma[2][2] * nx[2]); // 二次形式
                float bottom = sqrt(determinant(sigma));
                float top = exp(-0.5 * s2d) * ${((2.0 * Math.PI) ** -1.5).toFixed(10)};// 3 / 2 = 1.5
                float result = top / bottom;

                if (isNaN(result)) {
                    return 0.0;
                } else {
                    return result;
                }
            }

            void main() {
                if (gl_VertexID >= ${data_n * dist_n}) {
                    norm_x = zero;
                    return;
                }

                int n = gl_VertexID / ${dist_n}; // 横に連続している方がnorm_x_sumの計算でよい
                int m = gl_VertexID % ${dist_n}; // mは2の累乗である必要あり
                
                vec3 x_vec3 = getSampler2DVec3(x, n);

                norm_x = normdist(x_vec3, mu[m], sigma[m]);
            }
        `

        const norm_x_sum_shader = `
            uniform sampler2D norm_x;
            uniform float pi[${dist_n}];

            in float zero;
            out float norm_x_sum;

            void main() {
                int n = gl_VertexID;

                norm_x_sum = zero;

                if (n >= ${data_n}) {
                    return;
                }

                int x = (n % ${dist_n_texture_m}) * ${dist_n};
                int y = n / ${dist_n_texture_m};

                for(int k = 0; k < ${dist_n}; k++){
                    norm_x_sum += pi[k] * texelFetch(norm_x, ivec2(x + k, y), 0).r;
                }
            }
        `

        const gamma_shader = `
            uniform sampler2D norm_x;
            uniform sampler2D norm_x_sum;

            uniform float pi[${dist_n}];

            in float zero;
            out float gamma;

            void main() {
                if (gl_VertexID >= ${data_n * dist_n}) {
                    gamma = zero;
                    return;
                }

                int n = gl_VertexID / ${dist_n};
                int m = gl_VertexID % ${dist_n}; // data_nは1列に収まらないのでdist_nでまとめる

                int x = n % ${dist_n_texture_m};
                int y = n / ${dist_n_texture_m};

                float bottom = texelFetch(norm_x_sum, ivec2(x, y), 0).r;

                if (bottom > 0.0) {
                    gamma = pi[m] * (texelFetch(norm_x, ivec2(x * ${dist_n} + m, y), 0).r / bottom); // 列、行の順序で指定
                } else {
                    gamma = pi[m] / ${dist_n}.0;
                }
            }
        `;

        const gamma_sum_shader = `
            uniform sampler2D gamma;

            in float zero;
            out float gamma_sum;

            void main() {
                int m = gl_VertexID;

                gamma_sum = zero;

                for (int y = 0; y < ${dist_n_texture_h}; y++) {
                    for (int x = 0; x < ${dist_n_texture_w}; x += ${dist_n}) {
                        gamma_sum += texelFetch(gamma, ivec2(x + m, y), 0).r;
                    }
                }
            }
        `

        const sigma_shader = `
            uniform sampler2D x_data;
            uniform vec3 mu[${dist_n}];
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];
            
            in float zero;
            out vec3 sigma;
            
            vec3 getSampler2DVec3(sampler2D data, int n) {
                int x = (n % ${vec3_texture_m}) * 3;
                int y = n / ${vec3_texture_m};

                return vec3(texelFetch(data, ivec2(x, y), 0).r, texelFetch(data, ivec2(x + 1, y), 0).r, texelFetch(data, ivec2(x + 2, y), 0).r);
            }

            void main() {
                int m = gl_VertexID / 3;
                int k = gl_VertexID % 3;
                
                sigma = vec3(zero);

                for (int y = 0; y < ${dist_n_texture_h}; y++) {
                    for (int x = 0; x < ${dist_n_texture_m}; x++) {
                        float gamma_n_m = texelFetch(gamma, ivec2(x * ${dist_n} + m, y), 0).r;
                        vec3 x_vec3 = getSampler2DVec3(x_data, x + y * ${dist_n_texture_m});

                        sigma += gamma_n_m * vec3(
                            (x_vec3[k] - mu[m][k]) * (x_vec3[0] - mu[m][0]),
                            (x_vec3[k] - mu[m][k]) * (x_vec3[1] - mu[m][1]),
                            (x_vec3[k] - mu[m][k]) * (x_vec3[2] - mu[m][2])
                        );
                    }
                }

                sigma /= gamma_sum[m];
                
                // regularation_value倍した単位行列を加算
                sigma[k] += ${regularation_value.toFixed(10)};
            }
        `;

        // outにmat3を指定することは許されなかったのでsigmaは別に計算……
        const mu_pi_shader = `
            uniform sampler2D x_data;
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];

            in float zero;
            out vec3 mu;
            out float pi;

            vec3 getSampler2DVec3(sampler2D data, int n) {
                int x = (n % ${vec3_texture_m}) * 3;
                int y = n / ${vec3_texture_m};

                return vec3(texelFetch(data, ivec2(x, y), 0).r, texelFetch(data, ivec2(x + 1, y), 0).r, texelFetch(data, ivec2(x + 2, y), 0).r);
            }

            void main() {
                int m = gl_VertexID;

                mu = vec3(0.0, 0.0, 0.0);

                for (int y = 0; y < ${dist_n_texture_h}; y++) {
                    for (int x = 0; x < ${dist_n_texture_m}; x++) {
                        float gamma_n_m = texelFetch(gamma, ivec2(x * ${dist_n} + m, y), 0).r;
                        vec3 x_vec3 = getSampler2DVec3(x_data, x + y * ${dist_n_texture_m});

                        mu += gamma_n_m * x_vec3;
                    }
                }

                mu /= gamma_sum[m];
                pi = gamma_sum[m] / ${data_n}.0 + zero;
            }
        `;

        // Functions

        const log_p_func = (norm_x_sum: Float32Array) => {
            var log_p = 0;

            for (let i = 0; i < data_n; i++) {
                const log_norm = Math.log(norm_x_sum[i]);
                if (!isNaN(log_norm)) {
                    log_p += log_norm;
                }
            }

            return log_p;
        };

        // Variables

        const dist_n_texture_zero = new Float32Array(dist_n_texture_len);
        const data_n_texture_zero = new Float32Array(dist_n_texture_m * dist_n_texture_h);

        const dist_n_zero = new Float32Array(dist_n);
        const dist_n_vec3_zero = new Float32Array(dist_n * 3);

        const padding_x = new Float32Array(vec3_texture_len);
        padding_x.set(x, 0);

        const norm_x = new Float32Array(dist_n_texture_len);
        const norm_x_sum = new Float32Array(dist_n_texture_m * dist_n_texture_h);
        
        var log_p = -Infinity;
        var new_log_p;
        var incr = 1;

        const gamma = new Float32Array(dist_n_texture_len);
        const gamma_sum = new Float32Array(dist_n);

        const mu = init_mu.slice();
        const pi = init_pi.slice();
        const sigma = init_sigma.slice();

        // Parameters

        const norm_x_param = {
            id: 'norm_x_shader',
            vertexShader: norm_x_shader,
            args: {
                'x': gpgpu.makeTextureInfo('float', [vec3_texture_h, vec3_texture_w], padding_x),
                'mu': mu,
                'sigma': sigma,
                'zero': dist_n_texture_zero,
                'norm_x': norm_x
            }
        };

        console.log(dist_n_texture_h, dist_n_texture_w, norm_x.length);
        const norm_x_sum_param = {
            id: 'norm_x_sum_shader',
            vertexShader: norm_x_sum_shader,
            args: {
                'norm_x': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], norm_x),
                'pi': pi,
                'zero': data_n_texture_zero,
                'norm_x_sum': norm_x_sum
            }
        };

        const gamma_param = {
            id: 'gamma_shader',
            vertexShader: gamma_shader,
            args: {
                'norm_x': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], norm_x),
                'norm_x_sum': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_m], norm_x_sum),
                'pi': pi,
                'zero': dist_n_texture_zero,
                'gamma': gamma
            }
        };

        const gamma_sum_param = {
            id: 'gamma_sum_shader',
            vertexShader: gamma_sum_shader,
            args: {
                'gamma': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], gamma),
                'zero': dist_n_zero,
                'gamma_sum': gamma_sum
            }
        };

        const sigma_param = {
            id: 'sigma_shader',
            vertexShader: sigma_shader,
            args: {
                'x_data': gpgpu.makeTextureInfo('float', [vec3_texture_h, vec3_texture_w], padding_x),
                'mu': mu,
                'gamma': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], gamma),
                'gamma_sum': gamma_sum,
                'zero': dist_n_vec3_zero,
                'sigma': sigma
            }
        };

        const mu_pi_param = {
            id: 'mu_pi_shader',
            vertexShader: mu_pi_shader,
            args: {
                'x_data': gpgpu.makeTextureInfo('float', [vec3_texture_h, vec3_texture_w], padding_x),
                'gamma': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], gamma),
                'gamma_sum': gamma_sum,
                'zero': dist_n_zero,
                'mu': mu,
                'pi': pi
            }
        };

        // 1. norm_x, norm_sum
        gpgpu.compute(norm_x_param);
        gpgpu.compute(norm_x_sum_param);

        for (var i = 0; i < max_iteration_time; i++) {
            // 2. gamma, gamma_sum

            gpgpu.compute(gamma_param);
            gpgpu.compute(gamma_sum_param);

            //console.log(gamma);
            //console.log(gamma_sum);

            // 3. mu, pi, sigma

            gpgpu.compute(sigma_param);
            gpgpu.compute(mu_pi_param);

            //console.log(sigma);
            //console.log(mu);
            //console.log(pi);

            // 4. norm_x, norm_sum

            gpgpu.compute(norm_x_param);
            gpgpu.compute(norm_x_sum_param);

            //console.log(norm_x);
            //console.log(norm_x_sum);

            // 5. log_p, judge break

            new_log_p = log_p_func(norm_x_sum);
            incr = (new_log_p - log_p) / Math.max(Math.abs(new_log_p), 1);
            log_p = new_log_p;

            console.log(`finished ${i + 1} / ${max_iteration_time}, incr:${incr}, log_p:${log_p}`);

            if (incr < log_p_tolerance_value || isNaN(incr)) {
                break;
            }
        }

        gpgpu.clearAll();

        console.log(mu, dist_n);

        return {
            gmm: new GMM(dist_n, mu, pi, sigma),
            gamma: gamma,//.slice(0, data_n * dist_n),しない。
            gamma_sum: gamma_sum
        };
    }

    // データ点全体の平均と標準偏差を求める
    static EvalXMuStd(data_n: number, x: Float32Array): Float32Array[] {
        const x_sum = new Float32Array(3);
        const x_sum2 = new Float32Array(3);

        for (let i = 0; i < data_n; i++) {
            for (let j = 0; j < 3; j++) {
                x_sum[j] += x[i * 3 + j];
                x_sum2[j] += x[i * 3 + j] ** 2;
            }
        }

        const x_mu = Float32Array.from(x_sum, (value: number) => {return value / data_n});
        const x_mu2 = Float32Array.from(x_sum2, (value: number) => {return value / data_n});
        const x_std = new Float32Array(3);

        for (let i = 0; i < 3; i++) {
            x_std[i] = Math.sqrt(x_mu2[i] - x_mu[i] ** 2);
        }

        return [x_mu, x_std];
    }

    //　gammaから事後確率を計算
    static PostProbByGamma(dist_n: number, data_n: number, gamma: Float32Array): Float32Array {
        const gpgpu = GPGPU.CreateGPGPU();

        const max_texture_size = gpgpu.getMaxTextureSize();

        const dist_n_texture_m = Math.floor(max_texture_size / dist_n);
        const dist_n_texture_w = dist_n * dist_n_texture_m;
        const dist_n_texture_h = Math.ceil(data_n / dist_n_texture_m);
        const dist_n_texture_len = dist_n_texture_w * dist_n_texture_h;

        // Shaders
        const gamma_sum_shader = `
            uniform sampler2D gamma;
            
            in float zero;
            out float gamma_sum;

            void main() {
                int n = gl_VertexID;

                gamma_sum = zero;

                if (n >= ${data_n * dist_n}) {
                    return;
                }

                int x = (n % ${dist_n_texture_m}) * ${dist_n};
                int y = n / ${dist_n_texture_m};

                for (int m = 0; m < ${dist_n}; m++) {
                    gamma_sum += texelFetch(gamma, ivec2(x + m, y), 0).r;
                }
            }
        `

        const post_prob_shader = `
            uniform sampler2D gamma_sum;

            in float gamma;
            out float post_prob;

            void main() {
                int n = gl_VertexID / ${dist_n};

                if (n >= ${data_n}) {
                    post_prob = 0.0;
                    return;
                }

                int x = n % ${dist_n_texture_m};
                int y = n / ${dist_n_texture_m};
                
                post_prob = gamma / texelFetch(gamma_sum, ivec2(x, y), 0).r;
            }
        `;

        // Variables
        const data_n_zero = new Float32Array(dist_n_texture_m * dist_n_texture_h);
        const gamma_sum = new Float32Array(dist_n_texture_m * dist_n_texture_h);
        const post_prob = new Float32Array(dist_n_texture_len);

        // Parameters
        const gamma_sum_param = {
            id: 'gamma_sum_shader',
            vertexShader: gamma_sum_shader,
            args: {
                'gamma': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], gamma),
                'zero': data_n_zero,
                'gamma_sum': gamma_sum
            }
        }

        const post_prob_param = {
            id: 'post_prob_shader',
            vertexShader: post_prob_shader,
            args: {
                'gamma_sum': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_m], gamma_sum),
                'gamma': gamma,
                'post_prob': post_prob
            }
        }

        gpgpu.compute(gamma_sum_param);
        gpgpu.compute(post_prob_param);

        gpgpu.clear(gamma_sum_param.id);
        gpgpu.clear(post_prob_param.id);

        return post_prob; // sliceしない。
    }

    static ClusteringByPostProb(dist_n: number, data_n: number, post_prob: Float32Array): Uint32Array {
        const gpgpu = GPGPU.CreateGPGPU();
        
        const max_texture_size = gpgpu.getMaxTextureSize();

        const dist_n_texture_m = Math.floor(max_texture_size / dist_n);
        const dist_n_texture_w = dist_n * dist_n_texture_m;
        const dist_n_texture_h = Math.ceil(data_n / dist_n_texture_m);
//        const dist_n_texture_len = dist_n_texture_w * dist_n_texture_h;

        // Shaders
        const clustering_shader = `
            uniform sampler2D post_prob;

            in float zero;
            out float x_cluster;

            void main() {
                int n = gl_VertexID;
                
                float max_post_prob = zero;
                
                int x = (n % ${dist_n_texture_m}) * ${dist_n};
                int y = n / ${dist_n_texture_m};

                for (int m = 0; m < ${dist_n}; m++) {
                    float post_prob_n_m = texelFetch(post_prob, ivec2(x + m, y), 0).r;
                    
                    if (post_prob_n_m > max_post_prob) {
                        max_post_prob = post_prob_n_m;
                        x_cluster = float(m);
                    }
                }
            }
        `;

        // Variables
        const data_n_zero = new Float32Array(data_n);
        const x_cluster = new Float32Array(data_n);

        // Parameters
        const clustering_param = {
            id: 'clustering_shader',
            vertexShader: clustering_shader,
            args: {
                'post_prob': gpgpu.makeTextureInfo('float', [dist_n_texture_h, dist_n_texture_w], post_prob),
                'zero': data_n_zero,
                'x_cluster': x_cluster
            }
        }

        gpgpu.compute(clustering_param);
        gpgpu.clear(clustering_param.id);

        return new Uint32Array(x_cluster);
    }

    static DivideDataByCluster(
        dist_n: number, data_n: number, x: Float32Array, x_cluster: Uint32Array
    ): { divided_x: Float32Array[], divided_data_n: Uint32Array } {
        const divided_x = Array.from(new Array(dist_n), () => { return new Array(0) });
        const divided_data_n = new Uint32Array(dist_n);

        for (let n = 0; n < data_n; n++) {
            const m = Math.round(x_cluster[n]);

            divided_x[m].push(...x.slice(n * 3, n * 3 + 3));
            divided_data_n[m] += 1;
        }

        return {
            divided_x: Array.from(divided_x, (x) => { return new Float32Array(x) }),
            divided_data_n: divided_data_n
        };
    }

    // 変数を初期化する
    static InitMu(dist_n: number, data_n: number, x: Float32Array): Float32Array {
        const mu_x_cluster = kmeansInc(dist_n, data_n, x);
        const mu = mu_x_cluster[0];

        /*
        // あんまりよくない初期値
        for (let i = 0; i< dist_n; i++) {
            let norm_value = this.Rnorm();
            mu[0 + i * 3] = x_std[0] * norm_value + x_mu[0];
            mu[1 + i * 3] = x_std[1] * norm_value + x_mu[1];
            mu[2 + i * 3] = x_std[1] * norm_value + x_mu[2];
        }
        */

        return mu;
    }

    static InitPi(dist_n: number): Float32Array {
        return new Float32Array(dist_n).fill(1.0 / dist_n);
    }

    static InitSigma(dist_n: number, x_std: Float32Array): Float32Array {
        const sigma = new Float32Array(9 * dist_n).fill(0.0);

        for (let i = 0; i < dist_n; i++) {
            sigma[0 + i * 9] = x_std[0] ** 2;
            sigma[4 + i * 9] = x_std[1] ** 2;
            sigma[8 + i * 9] = x_std[2] ** 2;
        }

        return sigma;
    }

    // 標準正規分布の乱数(Box-Muller法)
    static Rnorm() {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}

/*// EvalXMuStd修正前
        const gpgpu = GPGPU.CreateGPGPU();

        const parsed_data_n = gpgpu.getMaxTextureSize();
        const _parseddata_n_n = Math.ceil(data_n / parsed_data_n);

        const x_mu_std_shader = `
            uniform sampler2D x;

            in vec3 x_sum;
            in vec3 x_sum2;
            out vec3 new_x_sum;
            out vec3 new_x_sum2;

            float getSampler2D(sampler2D data, int x, int y) {
                return texelFetch(data, ivec2(x, y), 0).r;
            }
    
            vec3 getSampler2DVec3(sampler2D data, int x) {
                return vec3(getSampler2D(data, 0, x), getSampler2D(data, 1, x), getSampler2D(data, 2, x));
            }

            void main() {
                new_x_sum = x_sum;
                new_x_sum2 = x_sum;
                
                for (int k = 0; k < ${parsed_data_n}; k++) {
                    vec3 x_vec3 = getSampler2D(x, k);

                    new_x_sum += x_vec3;
                    new_x_sum2 += vec3(x_vec3[0] * x_vec3[0], x_vec3[1] * x_vec3[1], x_vec3[2] * x_vec3[2]);
                }
            }
        `
*/
        /*
        const x_mu_std_param = {
            id: 'x_mu_std_shader',
            vertexShader: x_mu_std_shader,
            args: {
                'x': gpgpu.makeTextureInfo('float', [parsed_data_n, 3], x),
                'x_sum': x_mu,
                'x_sum2': x_std
            }
        }

        gpgpu.compute(x_mu_std_param);
        gpgpu.clear(x_mu_std_param.id);
        */