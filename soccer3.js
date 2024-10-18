
import { embed_text, getEmbbedder } from './embed.js';
import { crosswalk } from './crosswalk.js';
import { soc2010_6digit as soc2010 } from './common.js';
import * as ort from 'ort';

export class SOCcer3 {

    static version_info = new Map(Object.entries({
        "3.0.5": {
            "soccer_url": "./SOCcer_v3.0.5.onnx",
            "embedding_model_name": 'Xenova/GIST-small-Embedding-v0',
            "version": "3.0.5",
            "pooling": "cls",
            "soccerNetVersion": "0.0.1",
            "train_data": "May2024"
        },
        "3.0.6": {
            "soccer_url": "./SOCcer_v3.0.6.onnx",
            "embedding_model_name": 'Xenova/GIST-small-Embedding-v0',
            "version": "3.0.6",
            "pooling": "cls",
            "soccerNetVersion": "0.0.2",
            "train_data": "Oct2024"
        }
    }));

    constructor(version) {
        if (!SOCcer3.version_info.has(version)) {
            throw new Error(`Unknown version ${version}: allowed values ${[...SOCcer3.version_info.keys()]}`)
        }
        this.version = SOCcer3.version_info.get(version);
        this.ready = false;
        this.embedder = null;
        this.n = 10;

        console.log(`Loading SOCcerNet from: ${this.version.soccer_url}`)
        let session_promise = ort.InferenceSession.create(this.version.soccer_url);
        let embedder_promise = getEmbbedder(this.version)

        this.ready_promise = Promise.all([session_promise, embedder_promise])
        this.ready_promise.then(([session, embedder]) => {
            this.embedder = embedder;
            this.session = session;
            this.ready = true;
        })
    }

    async wait_until_ready() {
        if (!this.ready) {
            await this.ready_promise;
        }
    }

    async code_jobs(ids, JobTitles, JobTasks, crosswalk_info = null, k = 10) {
        await this.wait_until_ready()
        // embed the job info...
        let embeddings = await embed_text(ids, JobTitles, JobTasks, this.embedder, this.version.pooling)
        const embeddings_tensor = new ort.Tensor('float32', embeddings.data, embeddings.dims);

        // crosswalk the crosswalk info... set the default
        // in case there is not crosswalk info...
        let crosswalks = {
            data: new Float32Array(embeddings.dims[0] * 840),
            dims: [embeddings.dims[0], 840]
        }
        // if you give me crosswalk info.. do it now
        if (crosswalk_info && Object.keys(crosswalk_info).length > 0) {
            crosswalks = await crosswalk(crosswalk_info)
        }

        const crosswalk_tensor = new ort.Tensor('float32', crosswalks.data, crosswalks.dims);
        const feeds = {
            embedded_input: embeddings_tensor,
            crosswalked_inp: crosswalk_tensor
        }

        let results = await this.session.run(feeds);
        // convert to a 2d-array
        results = onnxResultToArray(results)
        // get the top k-results...
        results = results.map((row) => topK(row, k))

        return results
    }
}


export function topK(arr, k = 10) {
    // Set k to the length of the array if k is greater than the array length
    k = Math.min(k, arr.length);

    // Create an array of indices and sort it based on the values in arr
    const indices = Array.from(arr.keys()).sort((a, b) => arr[b] - arr[a]);

    // Get the top k values and their indices
    const topValues = indices.slice(0, k).map(i => arr[i]);
    const topIndices = indices.slice(0, k);
    const topLabels = topIndices.map(i => soc2010[i].title)
    const topCodes = topIndices.map(i => soc2010[i].soc_code)

    return { soc2010: topCodes, title: topLabels, score: topValues };
}


function onnxResultToArray(results) {
    const [rows, cols] = results.soc2010_out.dims;
    const data = Array.from(results.soc2010_out.cpuData);

    return Array.from({ length: rows }, (_, i) => data.slice(i * cols, i * cols + cols));
}
