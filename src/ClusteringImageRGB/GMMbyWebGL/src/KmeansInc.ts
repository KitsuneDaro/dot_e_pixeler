import * as GPGPU from './gpgpu';

export function kmeansInc(dist_n: number, data_n: number, x: Float32Array): Float32Array[] {
    /* k-means++ class (3D限定) */

    // Shaders
    const kmeans_inc_distance_shader = `
        uniform vec3 center;

        in vec3 x;
        in float distance2;
        out float new_distance2;

        void main(){
            vec3 delta = x - center;
            float temp_distance2 = dot(delta, delta);

            if (temp_distance2 < distance2) {
                new_distance2 = temp_distance2;
            } else {
                new_distance2 = distance2;
            }
        }
    `

    // Variables
    const distance2 = new Float32Array(data_n).fill(Infinity);

    const first_index = Math.floor(Math.random() * data_n);

    const centers = new Float32Array(3 * dist_n);
    const center = x.slice(first_index * 3, first_index * 3 + 3);
    centers.set(center, 0);

    // Parameters
    const kmeans_inc_distance_param = {
        id: 'kmeans_inc_distance_shader',
        vertexShader: kmeans_inc_distance_shader,
        args: {
            'center': center,
            'x': x,
            'distance2': distance2,
            'new_distance2': distance2
        }
    };

    const sum_func = (accumulator: number, currentValue: number) => {
        return accumulator + currentValue;
    }
/*
    const y = new Uint8ClampedArray(100 * 100 * 4);
    for (let i = 0; i < 100 * 100; i++) {
        y.set(x.slice(i * 3, (i + 1) * 3), i * 4);
        y[i * 4 + 3] = 255;
    }

    const p = document.createElement('canvas');
    ClusteringImageRGB.DrawCanvasRGBAData(100, 100, y, p);
    document.body.appendChild(p);
*/

    for (let k = 1; k < dist_n; k++) {
        // distance2を計算
        GPGPU.gpgpu.compute(kmeans_inc_distance_param);

        // indexを累積和から計算
        let index = data_n - 1; // 浮動小数点によってdistance2の和が1.0にならなかったときのための初期値
/*
        const y = new Uint8ClampedArray(100 * 100 * 4);
        for (let i = 0; i < 100 * 100; i++) {
            y.set(distance2.slice(i * 3, (i + 1) * 3), i * 4);
            y[i * 4 + 3] = 255;
        }

        const p = document.createElement('canvas');
        ClusteringImageRGB.DrawCanvasRGBAData(100, 100, y, p);
        document.body.appendChild(p);
*/
        let distance2_sum = distance2.reduce(sum_func);

        if (distance2_sum == 0.0) {
            throw new Error('分布数に対してデータに含まれるベクトルの値の種類が少なすぎます。');
        }

        let distance2_cumsum = 0;
        let random = Math.random() * distance2_sum;

        for (var i = 0; i < data_n; i++) {
            distance2_cumsum += distance2[i];

            if (random < distance2_cumsum) {
                index = i;
                break;
            }
        }

        // centerに求めた点を追加
        center.set(x.slice(index * 3, index * 3 + 3), 0);
        centers.set(center, k * 3);
    }

    GPGPU.gpgpu.clear(kmeans_inc_distance_param.id);

    return kmeans(dist_n, data_n, x, centers);
}

export function kmeans(dist_n: number, data_n: number, x: Float32Array, init_centers: Float32Array): Float32Array[] {
    /* k-means class (3D限定) */

    const max_texture_size = GPGPU.gpgpu.getMaxTextureSize();

    const vec3_texture_m = Math.floor(max_texture_size / 3);
    const vec3_texture_w = 3 * vec3_texture_m;
    const vec3_texture_h = Math.ceil(data_n / vec3_texture_m);
    const vec3_texture_len = vec3_texture_w * vec3_texture_h;

    // Shaders
    const kmeans_clustering_shader = `
        uniform vec3 centers[${dist_n}];

        in vec3 x;
        out float x_cluster; // 0 ~ dist_n - 1

        void main() {
            if (gl_VertexID >= ${data_n}) {
                x_cluster = -1.0;
                return;
            }

            x_cluster = 0.0;

            vec3 delta = x - centers[0];
            float min_distance2 = dot(delta, delta);

            for (int k = 1; k < ${dist_n}; k++) {
                vec3 delta = x - centers[k];
                float distance2 = dot(delta, delta);

                if (distance2 < min_distance2) {
                    min_distance2 = distance2;
                    x_cluster = float(k);
                }
            }
        }
    `;

    const kmeans_centers_shader = `
        uniform sampler2D x_data;
        uniform sampler2D x_cluster;

        in float zero;
        out vec3 center;
        out float x_cluster_n;

        vec3 getSampler2DVec3(sampler2D data, int n) {
            int x = (n % ${vec3_texture_m}) * 3;
            int y = n / ${vec3_texture_m};

            return vec3(texelFetch(data, ivec2(x, y), 0).r, texelFetch(data, ivec2(x + 1, y), 0).r, texelFetch(data, ivec2(x + 2, y), 0).r);
        }

        void main() {
            int k = 0;
            int i = 0;
            
            center = vec3(0.0, 0.0, 0.0);
            x_cluster_n = zero;

            for (int y = 0; y < ${vec3_texture_h}; y++) {
                for (int x = 0; x < ${vec3_texture_m}; x++) {
                    i = x + y * ${vec3_texture_m};

                    k = int(texelFetch(x_cluster, ivec2(x, y), 0).r);

                    if (k == gl_VertexID) {
                        center += getSampler2DVec3(x_data, i);
                        x_cluster_n += 1.0;
                    }
                }
            }

            center /= x_cluster_n;
        }
    `;

    // Variables
    const dist_n_zero = new Float32Array(dist_n);

    const padding_x = new Float32Array(vec3_texture_len);
    padding_x.set(x, 0);

    const x_cluster = new Float32Array(vec3_texture_m * vec3_texture_h);
    const x_cluster_n = new Float32Array(dist_n);

    const centers = init_centers.slice();
    const new_centers = new Float32Array(dist_n * 3);

    // Parameters
    const kmeans_clustering_param = {
        id: 'kmeans_clustering_shader',
        vertexShader: kmeans_clustering_shader,
        args: {
            'centers': centers,
            'x': padding_x,
            'x_cluster': x_cluster
        }
    };

    const kmeans_centers_param = {
        id: 'kmeans_centers_shader',
        vertexShader: kmeans_centers_shader,
        args: {
            'x_data': GPGPU.gpgpu.makeTextureInfo('float', [vec3_texture_h, vec3_texture_w], padding_x),
            'x_cluster': GPGPU.gpgpu.makeTextureInfo('float', [vec3_texture_h, vec3_texture_m], x_cluster),
            'zero': dist_n_zero,
            'x_cluster_n': x_cluster_n,
            'center': new_centers
        }
    };

    for (let k = 0; k < 1; k++) {

        GPGPU.gpgpu.compute(kmeans_clustering_param);
        GPGPU.gpgpu.compute(kmeans_centers_param);

        var flag = true;

        for (let k = 0; k < dist_n; k++) {
            var cluster_n = x_cluster_n[k];

            for (let i = k * 3; i < (k + 1) * 3; i++) {
                if (cluster_n > 0) {
                    if (new_centers[i] != centers[i]) {
                        flag = false;
                    }
                } else {
                    new_centers[i] = centers[i];
                }
            }
        }

        if (flag) {
            break;
        }

        centers.set(new_centers.slice(), 0);
    }

    GPGPU.gpgpu.clear(kmeans_clustering_param.id);
    GPGPU.gpgpu.clear(kmeans_centers_param.id);

    return [centers, x_cluster.slice(0, data_n)];
}