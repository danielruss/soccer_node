import { pipeline } from '@xenova/transformers';
import * as tf from '@tensorflow/tfjs'
import { abbrev as abbreviation } from './abbrev.js'


/*********************************************************
 * crosswalking
 *********************************************************/
const knownCrosswalks = new Map([
    ["soc1980", "https://danielruss.github.io/codingsystems/soc1980_soc2010.json"],
    ["noc2011", "https://danielruss.github.io/codingsystems/noc2011_soc2010_via_soc2018.json"],
    ["soc2018", "https://danielruss.github.io/codingsystems/soc2010_soc2018.json"]
])
export const availableCodingSystems = Array.from(knownCrosswalks.keys());


async function buildCrossWalk(url, system) {
    const raw = await (await fetch(url)).json()
    const xw = raw.reduce((acc, current) => {
        if (!acc.has(current[system])) {
            acc.set(current[system], [])
        }
        acc.get(current[system]).push(current['soc2010'])
        return acc;
    }, new Map())
    return xw
}

const crosswalks = new Map();
knownCrosswalks.forEach(async (url, system) => {
    crosswalks.set(system, await buildCrossWalk(url, system))
})


export async function crosswalk(system, codes) {
    // this inner function crosswalks 1 code to 
    // a multi-hot encoding.  xw is the crosswalk
    // code is the code we are crosswalking to soc2010
    // index is the id of the job.
    function xw_one(xw, code, index, buffer) {
        let soc2010_codes = xw.get(code)
        if (!soc2010_codes) return
        soc2010_codes.forEach(soc_code => {
            let info = soc2010_6digit.get(soc_code)
            buffer.set(1., index, info.index)
        })
    }

    if (!knownCrosswalks.has(system)) throw new Error(`Unknow coding system: ${system}`)
    const xw = crosswalks.get(system);
    if (!Array.isArray(codes)) codes = [codes]

    let mhe = tf.buffer([codes.length, 840])
    // for each job. cross walk the other code.
    // if there are multiple codes for a job, get all the codes.
    codes.map((code, index) => {
        if (Array.isArray(code)) {
            code.forEach(cd => xw_one(xw, cd, index, mhe))
        } else {
            xw_one(xw, code, index, mhe)
        }
    })
    return mhe.toTensor()
}

/*********************************************************
 * SOC 6-digit
 *********************************************************/
async function get_soc2010() {
    const soc2010 = (await (await fetch("https://danielruss.github.io/codingsystems/soc2010_6digit.json")).json())
        .filter(x => x.soc_code != "99-9999")
    const soc2010_map = new Map()
    for (let i = 0; i < soc2010.length; i++) {
        const current_soc = {
            index: i,
            code: soc2010[i].soc_code,
            title: soc2010[i].title
        }
        soc2010_map.set(current_soc.code, current_soc)
    }
    return soc2010_map
}
const soc2010_6digit = await get_soc2010()





// I may need to shove this is github and read from there...
//let abbrev_path = /^\/test/.test(location.pathname)?"../abbrev.json":"/abbrev.json"
//let abbrev_path = "http://localhost:8000/abbrev.json"
//let abbrev_path = "./abbrev.json"
//let abbrev = await (await fetch(abbrev_path)).json()
const abbrev_keys = Object.keys(abbreviation)
let abbrev = Object.fromEntries(Object.entries(abbreviation).filter(([key, value]) => !Array.isArray(value)))

await tf.ready()


export class SOCcer3 {

    // init parameters...
    #init_promise = null;
    #ready = false;

    #callback = null;
    #pipeline = null;
    #soccer = null;
    #version = null;
    #abbrev = null;

    n = 10;

    static version_info = new Map(Object.entries({
        "3.0.1": {
            "soccer_url": "https://danielruss.github.io/soccer-models/s_v3.0.1/model.json",
            "embedding_model_name": "Xenova/all-MiniLM-L6-v2",
            "version": "3.0.1",
            "pooling": "mean"
        },
        "3.0.2": {
            "soccer_url": "https://danielruss.github.io/soccer-models/s_v3.0.2.tfjs/model.json",
            "embedding_model_name": "Xenova/all-MiniLM-L6-v2",
            "version": "3.0.2",
            "pooling": "mean"
        },
        "3.0.3": {
            "soccer_url": "https://danielruss.github.io/soccer-models/s_v3.0.3.tfjs/model.json",
            "embedding_model_name": "Xenova/GIST-small-Embedding-v0",
            "version": "3.0.3",
            "pooling": "cls"
        },
        "3.0.4": {
            "soccer_url": "https://danielruss.github.io/soccer-models/s_v3.0.4.tfjs/model.json",
            "embedding_model_name": "Xenova/GIST-small-Embedding-v0",
            "version": "3.0.4",
            "train_data": "May2024",
            "pooling": "cls"
        }
    }));

    constructor(version) {
        if (!SOCcer3.version_info.has(version)) {
            throw new Error(`Unknown version ${version}: allowed values ${[...SOCcer3.version_info.keys()]}`)
        }
        this.#version = SOCcer3.version_info.get(version);

        this.#init_promise = this.initalize_soccer3()
        this.#init_promise.then(() => {
            this.#ready = true
        })
            .catch((error) => {
                throw error
            })
    }

    get ready() {
        return this.#ready
    }

    get version() {
        return this.#version.version
    }

    set n(value) {
        this.n = parseInt(value)
    }

    async embed_text(text) {
        console.time("time to embed block")
        let emb_obj = await this.#pipeline(text, {
            pooling: this.#version.pooling,
            normalize: true,
        })
        console.timeEnd("time to embed block")
        const res = await tf.tidy(() => {
            return tf.reshape(tf.tensor(emb_obj.data), emb_obj.dims)
        })
        return res;
    }

    async wait_until_ready() {
        if (!this.ready) {
            await this.#init_promise;
        }
    }

    async code(JobTitleTask, soc2010_xw_tensor) {
        await this.wait_until_ready()
        let emb_jtt = await this.embed_text(JobTitleTask)
        let res = this.#soccer.predict([emb_jtt, soc2010_xw_tensor])
        emb_jtt.dispose()
        soc2010_xw_tensor.dispose()
        return res
    }

    // pass in an array of JobTitles, JobTasks, and
    async code_jobs(id, jobTitle, jobTask, crosswalk_object = {}) {
        await this.wait_until_ready()
        let fields = ["Id", "JobTitle", "JobTask"]

        // convert the input into arrays if they are not....
        jobTitle = Array.isArray(jobTitle) ? jobTitle : [jobTitle]
        jobTask = Array.isArray(jobTask) ? jobTask : [jobTask]
        if (jobTitle.length != jobTask.length) {
            return { error: "Error in code_jobs: # of JobTitles != # of JobTasks" }
        }
        if (!id) id = new Array(jobTitle.length).fill(null).map((_, i) => `row-${i + 1}`);
        id = Array.isArray(id) ? id : [id]
        // preprocess
        let preprocessed_data = preprocess(id, jobTitle, jobTask)

        // crosswalk
        let xw_tensor = tf.zeros([jobTitle.length, 840])
        for (const [key, value] of Object.entries(crosswalk_object)) {
            let codes = Array.isArray(value) ? value : [value]
            let ok = codes.length == jobTitle.length
            if (knownCrosswalks.has(key) && ok) {
                fields.push(key)
                preprocessed_data.forEach(row => row[key] = value)
                let current_xw = await crosswalk(key, codes);
                let new_tensor = tf.maximum(xw_tensor, current_xw);
                tf.dispose([xw_tensor, current_xw]);
                xw_tensor = new_tensor;
            }
        }

        // get the JobTitleTask, but remove it from the
        // preprocessed data.  This will be returned to the 
        // user.
        let jobTitleTask = preprocessed_data.map(row => {
            const v = row.JobTitleTask
            delete row.JobTitleTask
            return v
        });

        // code the job
        xw_tensor.print()

        // the xw_tensor is dispose in the code method...
        let res = await this.code(jobTitleTask, xw_tensor)
        res = this.analyze_chunk(res);
        res.input = preprocessed_data;
        res.fields = fields

        tf.disposeVariables();
        console.log(`number of tensors in memory: ${tf.memory().numTensors}`)
        return res
    }

    // TO DO: This should call code_jobs(JobTitle,JobTask,XW) -> Codes/Titles/Scores
    //        in a tf.tidy() to prevent memory leaks.
    // TO DO: There should also be a code_job(JobTitle,JobTask,XW) that makes an
    //        array out of the title,task,xw and calls code_chunk.
    async code_papa_chunk(results) {
        tf.engine().startScope();

        // preprocess the data...
        let preprocessed_data = preprocess_papa_chunk(results);

        let fields = [...results.meta.fields]
        if (!(fields.includes("Id") || fields.includes("id"))) {
            fields.unshift("Id")
        }

        // crosswalk soc1980, noc2011, soc2018
        // if no crosswalk, just pass in zeros...
        let xw_tensor = tf.zeros([results.data.length, 840])
        if (fields.some(col => knownCrosswalks.has(col))) {
            let xw_obj = {}
            knownCrosswalks.forEach((value, key) => xw_obj[key] = [])
            for (const row of results.data) {
                xw_obj.soc1980.push(row?.soc1980)
                xw_obj.noc2011.push(row?.noc2011)
                xw_obj.soc2018.push(row?.soc2018)
            }
            // crosswalk any coded column
            for (const key of knownCrosswalks.keys()) {
                if (fields.includes(key)) {
                    let soc2010_xw = await crosswalk(key, xw_obj[key])
                    xw_tensor = tf.maximum(xw_tensor, soc2010_xw)
                    tf.dispose(soc2010_xw)
                }
            }
        }
        let jobTitleTask = preprocessed_data.map(row => row.JobTitleTask)
        // the xw_tensor is disposed of in the code method
        let res = await this.code(jobTitleTask, xw_tensor)
        res = this.analyze_chunk(res)
        res.input = preprocessed_data
        res.fields = fields

        tf.engine().endScope();
        return res
    }

    analyze_chunk(tensor) {
        let soc2010_array = Array.from(soc2010_6digit.keys())
        let sorted_results = tf.topk(tensor, this.n ?? 10)
        tensor.dispose()
        let codes = sorted_results.indices.arraySync().map(row => row.map(v => soc2010_array[v]))
        let titles = codes.map(row => row.map(c => soc2010_6digit.get(c).title))
        let scores = sorted_results.values.arraySync()
        sorted_results.indices.dispose()
        sorted_results.values.dispose()

        return {
            codes: codes,
            titles: titles,
            scores: scores
        }
    }

    async initalize_soccer3() {
        this.#pipeline = await pipeline('feature-extraction', this.#version.embedding_model_name, { quantized: false })
        this.#soccer = await tf.loadLayersModel(this.#version.soccer_url);

        //let abbrev_path = /^\/test/.test(location.pathname)?"../abbrev.json":"abbrev.json"
        //let abbrev = await (await fetch(abbrev_path)).json()
        const abbrev_keys = Object.keys(abbreviation)
        this.#abbrev = Object.fromEntries(Object.entries(abbreviation).filter(([key, value]) => !Array.isArray(value)))
    }
}

function preprocess_one_line(args) {
    // handle NA/NaN/empty arguments
    args.JobTitle ??= " ";
    args.JobTask ??= " ";

    // clean the JobTitle/JobTask...
    args.JobTitle = args.JobTitle
        .replaceAll(/^[\s\-\.]+|[\s\-\.]+$/g, "")
        .toLowerCase()
    args.JobTask = args.JobTask
        .replaceAll(/^[\s\-\.]+|[\s\-\.]+$/g, "")
        .toLowerCase()
    // handle abbreviations...
    args.JobTitle = Object.hasOwn(abbrev, args.JobTitle) ? abbrev[args.JobTitle] : args.JobTitle
    args.JobTask = Object.hasOwn(abbrev, args.JobTask) ? abbrev[args.JobTask] : args.JobTask

    // combine the job title and job task
    args.JobTitleTask = `${args.JobTitle} ${args.JobTask}`.trim();
    return args;
}

export function preprocess(id, jobTitles, jobTasks) {
    let preprocessed_data = jobTitles.map((title, index) => {
        let row = preprocess_one_line({ JobTitle: title, JobTask: jobTasks[index] })
        row.Id = id[index]
        return row;
    })
    return preprocessed_data
}

function preprocess_papa_chunk(results) {

    let preprocessed_data = results.data.map((row, indx) => {
        row = preprocess_one_line(row)
        row.Id = row.Id ?? `row-${(indx + results.meta.start_row + 1).toString().padStart(6, '0')}`
        return row;
    })
    return preprocessed_data
}