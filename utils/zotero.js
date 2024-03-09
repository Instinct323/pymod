// https://www.zotero.org/support/dev/client_coding/javascript_api/search_fields
const fields = [
    "publicationTitle",     // 期刊
    "journalAbbreviation",  // 期刊缩写
    "conferenceName",       // 会议名称
    "proceedingsTitle",     // 会议论文集
    "publisher",            // 出版社
    "bookTitle",            // 书名
    "repository",           // 仓库
    "title", "volume", "pages", "date", "url", "extra",
    "issue", "ISSN", "ISBN", "abstractNote", "place", "libraryCatalog"
]


// 弹出消息框
function log(msg, level = 0, duration = 5000) {
    const levels = ["INFO", "WARN", "ERROR", "FATAL"];
    let popw = new Zotero.ProgressWindow();

    popw.changeHeadline(levels[level]);
    popw.addDescription(msg);
    popw.show();
    popw.startCloseTimer(duration);
}


async function getDoi(item) {
    let doi = item.getField("DOI");
    if (!doi) {
        // 从 extra 字段中提取
        let extra = item.getField("extra");
        let regex = /DOI:\s*(\S+)/;
        let match = extra.match(regex);
        if (match) {
            doi = match[1];
            try {
                item.setField("DOI", match[1]);
                await item.saveTx();
            } catch (e) {
            }
        }
    }
    return doi;
}


// 填充 arxiv 的 DOI
async function fillArxivDoi(item) {
    const url = item.getField("url");
    const pat = new RegExp("https?://arxiv\.org/abs/");
    let is_arxiv = !!url.match(pat);

    if (is_arxiv && !(await getDoi(item))) {
        let arxiv_id = url.split("/").pop();
        let arxiv_doi = "10.48550/arXiv." + arxiv_id;
        // 覆盖写入 DOI
        item.setField("DOI", arxiv_doi);
        log(url + " -> " + arxiv_doi);
    }
    return is_arxiv;
}


// 合并元数据
async function mergeMetadata(item, newItem, cover = true) {
    // Fields: https://www.zotero.org/support/dev/client_coding/javascript_api/search_fields
    let etype = [];
    item.setCreators(newItem.getCreators());    // 作者
    // 覆盖元数据
    for (let field of fields) {
        if (cover || newItem.getField(field)) {
            try {
                item.setField(field, newItem.getField(field));
            } catch (e) {
                etype.push(field);
            }
        }
    }
    // 输出错误信息
    let msg = "";
    for (let field of etype) {
        msg += field + ": " + newItem.getField(field) + "\n";
    }
    item.setField("extra", newItem.getField("extra") + "\n" + msg);
    // 清理副本
    newItem.deleted = true;
    await newItem.saveTx();
    return etype.length;
}


class MetadataUpdater {

    constructor(items, pool_size = 4) {
        this.items = items;
        this.tags = ["🍋 Queue", "🥥 No DOI found", "🍓 Type error", "🥝 Ignore"];
        this.cnt = [0, 0, 0, 0];
    }

    info() {
        let msg = "MAINTAINER: CSDN @ 荷碧TongZJ\n";
        for (let i = 0; i < this.tags.length; i++) {
            msg += this.tags[i] + ": " + this.cnt[i] + ", ";
        }
        if (this.cnt[0] === 0) {
            msg += "Done!";
        }
        log(msg);
    }

    async run() {
        const cn_char = /[\u4e00-\u9fa5]/;
        // 初始化
        for (let item of this.items) {
            item.addTag(this.tags[0]);
            this.cnt[0]++;
        }
        this.info();
        for (let item of this.items) {
            let title = item.getField("title");
            // 英文标题, 处理
            if (!title.match(cn_char)) {
                await this.process(item);
            } else {
                // 中文标题, 忽略
                item.addTag(this.tags[3]);
                this.cnt[3]++;
                await item.saveTx();
            }
            item.removeTag(this.tags[0]);
            this.cnt[0]--;
            this.info();
        }
    }

    async process(item) {
        let is_arxiv = await fillArxivDoi(item);
        await item.saveTx()
        // 更新元数据
        if (await getDoi(item)) {
            item.removeTag(this.tags[1]);
            await this.updateMetadata(item);
        } else {
            item.addTag(this.tags[1]);
            this.cnt[1]++;
        }
        // 填充 arxiv 仓库
        if (is_arxiv) {
            item.setField("repository", "arXiv");
        }
        await item.saveTx();
    }

    async updateMetadata(item) {
        // 期刊论文模板
        let translate = new Zotero.Translate.Search();
        translate.setIdentifier({
            itemType: "journalArticle",
            DOI: await getDoi(item)
        });
        translate.setTranslator(await translate.getTranslators());
        // 覆盖式合并
        let newItem = (await translate.translate())[0];
        if (newItem.getField("title") && await mergeMetadata(item, newItem, true)) {
            item.addTag(this.tags[2]);
            this.cnt[2]++;
        } else {
            item.removeTag(this.tags[2]);
        }
    }
}


async function clearMetadata() {
    // 清空元数据
    let cnt = 0;
    for (let item of items) {
        if (await getDoi(item)) {
            for (let field of fields) {
                if (field !== "title" && field !== "extra") {
                    item.setField(field, "");
                }
            }
            await item.saveTx();
        }
        cnt++;
        if (!(cnt % 10)) {
            log("clear metadata: " + cnt);
        }
    }
    log("clear metadata: Done!");
}


var items = Zotero.getActiveZoteroPane().getSelectedItems();
let mu = new MetadataUpdater(items);
await mu.run();
// await clearMetadata();
