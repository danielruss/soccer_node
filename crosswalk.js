import { soc2010_info } from './common.js';

const calcIndex = (row, col) => row * 840 + col
const knownCrosswalks = new Map([
    ["soc1980", "https://danielruss.github.io/codingsystems/soc1980_soc2010.json"],
    ["noc2011", "https://danielruss.github.io/codingsystems/noc2011_soc2010_via_soc2018.json"],
    ["isco1988", "https://danielruss.github.io/codingsystems/isco1988_soc2010.json"],
])
export const availableCodingSystems = Array.from(knownCrosswalks.keys());

let crosswalk_cache = new Map()

async function buildCrossWalk(url, system) {
    // node doesn't have localforage.. use a map for cacheing ...
    if (crosswalk_cache.has(system)) {
        return crosswalk_cache.get(system)
    }

    const raw = await (await fetch(url)).json()
    const xw = raw.reduce((acc, current) => {
        if (!acc.has(current[system])) {
            acc.set(current[system], [])
        }
        acc.get(current[system]).push(current['soc2010'])
        return acc;
    }, new Map())

    // add the new xw to the cache.
    crosswalk_cache.set(system, xw)
    return xw
}

const promises = [...knownCrosswalks.entries()].map(([system, url]) => buildCrossWalk(url, system));
await Promise.all(promises)


export async function crosswalk(crosswalk_object) {
    let res = null;
    for (let k of knownCrosswalks.keys()) {
        if (Object.hasOwn(crosswalk_object, k)) {
            let r1 = await crosswalk_from(k, crosswalk_object[k])
            if (!res) {
                res = r1
            } else {
                for (let i = 0; i < res.data.length; i++) {
                    res.data[i] = Math.max(res.data[i], r1.data[i])
                }
            }
        }
    };
    return res;
}

async function crosswalk_from(system, codes) {
    // this inner function crosswalks 1 code to 
    // a multi-hot encoding.  xw is the crosswalk
    // code is the code we are crosswalking to soc2010
    // index is the id of the job.
    function xw_one(xw, code, index, buffer) {
        let soc2010_codes = xw.get(code)
        if (!soc2010_codes) return
        soc2010_codes.forEach(soc_code => {
            let info = soc2010_info.get(soc_code)
            let buffer_index = calcIndex(index, info.index)
            buffer[buffer_index] = 1.
        })
    }

    if (!knownCrosswalks.has(system)) throw new Error(`Unknow coding system: ${system}`)
    const xw = await buildCrossWalk(knownCrosswalks.get(system), system);
    if (!Array.isArray(codes)) codes = [codes]

    //let mhe = tf.buffer([codes.length, 840])
    // TO DO: remove magic 840 (soc2010 specific)...
    let mhe = new Float32Array(codes.length * 840)
    let dims = [codes.length, 840]


    // for each job. cross walk the other code.
    // if there are multiple codes for a job, get all the codes.
    codes.map((code, index) => {
        if (Array.isArray(code)) {
            code.forEach(cd => xw_one(xw, cd, index, mhe))
        } else {
            xw_one(xw, code, index, mhe)
        }
    })
    return { data: mhe, dims: dims }
}