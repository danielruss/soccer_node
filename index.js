import { SOCcer3 } from "./soccer3.js";


export const helloGET = (req, res) => {
    console.log(`=====> Received request for:`, req.params[0])
    switch (req.params[0]) {
        case "":
            heartbeat(req, res);
            break;
        case "code":
            code(req, res)
            break;
        default:
            console.log(`could not handle `, req.params[0])
            res.sendStatus(404);
    }
};

function heartbeat(req, res) {
    res.send(`The time in downtown here is: ${new Date()}`);
}

async function code(req, res) {
    try {
        let x = await (await fetch("https://danielruss.github.io/codingsystems/soc1980_soc2010.json")).json()
        let id = "s3api"
        let soccer304 = new SOCcer3("3.0.4")
        let jobTitle = req.query.title ?? ""
        let jobTask = req.query.task ?? ""
        let n = req.query.n ?? 10
        if (Array.isArray(n)) n = n[0];
        n = Math.min(Math.max(1, n), 840)
        soccer304.n = n;

        let xw_obj = {}
        if (req.query.soc1980) xw_obj.soc1980 = req.query.soc1980
        if (req.query.noc2011) xw_obj.noc2011 = req.query.noc2011
        if (req.query.soc2018) xw_obj.soc2018 = req.query.soc2018
        let soccer_res = await soccer304.code_jobs(id, jobTitle, jobTask, xw_obj)
        let outputs = {
            jobTitle: jobTitle,
            jobTask: jobTask,
            crosswalk_info: xw_obj,
            soccer: [],
            soccer_version: "3.0.4",
            date: new Date()
        }
        for (let rank = 0; rank < n; rank++) {
            outputs.soccer.push({
                rank: rank + 1,
                code: soccer_res.codes[0][rank],
                title: soccer_res.titles[0][rank],
                score: soccer_res.scores[0][rank],
            });
        }
        res.setHeader('Content-Type', 'application/json');
        res.send(JSON.stringify(outputs))
    } catch (error) {
        console.error(error)
    }
}