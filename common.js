export const soc2010_6digit = (await (await fetch("https://danielruss.github.io/codingsystems/soc2010_6digit.json")).json())
    .filter(x => x.soc_code != "99-9999")

export const soc2010_info = new Map()
soc2010_6digit.forEach((code, index) => soc2010_info.set(code.soc_code, {
    code: code.soc_code,
    title: code.title,
    index: index
}))

