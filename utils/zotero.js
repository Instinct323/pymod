// https://www.zotero.org/support/dev/client_coding/javascript_api/search_fields
const FIELDS = [
    "publicationTitle",     // 期刊
    "journalAbbreviation",  // 期刊缩写
    "conferenceName",       // 会议名称
    "proceedingsTitle",     // 会议论文集
    "bookTitle",            // 书名
    "publisher",            // 出版社
    "repository",           // 仓库
    "title", "volume", "pages", "date", "url", "extra",
    "issue", "ISSN", "ISBN", "abstractNote", "place", "libraryCatalog"
]
const ITEMS = Zotero.getActiveZoteroPane().getSelectedItems();


// 弹出消息框
function log(msg, level = 0, duration = 5000) {
    const levels = ["INFO", "WARN", "ERROR", "FATAL"];
    let popw = new Zotero.ProgressWindow();

    popw.changeHeadline(levels[level]);
    popw.addDescription(msg);
    popw.show();
    popw.startCloseTimer(duration);
}


// 提取 DOI
async function getDoi(item) {
    let doi = item.getField("DOI");
    if (!doi) {
        // 从 extra 字段中提取
        const regex = /DOI:\s*(\S+)/;
        const match = item.getField("extra").match(regex);
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
    const is_arxiv = !!url.match(pat);

    if (is_arxiv && !(await getDoi(item))) {
        const arxiv_id = url.split("/").pop();
        const arxiv_doi = "10.48550/arXiv." + arxiv_id;
        // 覆盖写入 DOI
        item.setField("DOI", arxiv_doi);
        await item.saveTx();
        log(url + " -> " + arxiv_doi);
    }
    return is_arxiv;
}


// 通过 DOI 加载元数据
async function loadMetadata(doi) {
    let translate = new Zotero.Translate.Search();
    translate.setIdentifier({
        itemType: "journalArticle",
        DOI: doi
    });
    translate.setTranslator(await translate.getTranslators());
    try {
        return (await translate.translate())[0];
    } catch (e) {
        return false;
    }
}


// 合并元数据
async function mergeMetadata(item, newItem, cover = false) {
    item.setCreators(newItem.getCreators());    // 作者
    const etype = FIELDS.filter(field => {
        if (!(cover || newItem.getField(field))) {
            return false;
        }
        // 覆盖元数据
        try {
            item.setField(field, newItem.getField(field));
            return false;
        } catch (e) {
            if (field !== "bookTitle") {
                return true;
            }
            // 特殊处理: bookTitle -> conferenceName, proceedingsTitle
            try {
                item.setField("conferenceName", newItem.getField("bookTitle"));
                item.setField("proceedingsTitle", newItem.getField("bookTitle"));
                return false;
            } catch (e) {
                return true;
            }
        }
    });
    // 输出错误信息
    let msg = "";
    for (let field of etype) {
        msg += field + ": " + newItem.getField(field) + "\n";
    }
    item.setField("extra", newItem.getField("extra") + "\n" + msg);
    return etype.length;
}


class MetadataUpdater {

    constructor() {
        this.tags = ["🍋 Queue", "🥥 No DOI found", "🍑 Type error", "🍓 Fail", "🥝 Ignore"];
        this.cnt = new Array(this.tags.length).fill(0);
    }

    info() {
        let msg = "MAINTAINER: CSDN @ 荷碧TongZJ\n";
        for (let i = 0; i < this.tags.length; i++) {
            msg += this.tags[i] + ": " + this.cnt[i] + ", \n";
        }
        if (!this.cnt[0]) {
            msg += "Done!";
        }
        log(msg);
    }

    async run() {
        const cn_char = /[\u4e00-\u9fa5]/;
        // 初始化
        log("Initializing...")
        for (let item of ITEMS) {
            item.addTag(this.tags[0]);
            this.cnt[0]++;
            await item.saveTx();
        }
        this.info();
        for (let item of ITEMS) {
            // 英文标题, 处理
            if (!item.getField("title").match(cn_char)) {
                await this.process(item);
            } else {
                // 中文标题, 忽略
                item.addTag(this.tags[4]);
                this.cnt[4]++;
            }
            item.removeTag(this.tags[0]);
            this.cnt[0]--;
            await item.saveTx();
            this.info();
        }
    }

    async process(item) {
        const is_arxiv = await fillArxivDoi(item);
        // 更新元数据
        if (await getDoi(item)) {
            item.removeTag(this.tags[1]);
            let newItem = await loadMetadata(await getDoi(item));
            if (newItem) {
                item.removeTag(this.tags[3]);
                // 覆盖式合并
                if (newItem.getField("title") && await mergeMetadata(item, newItem, true)) {
                    item.addTag(this.tags[2]);
                    this.cnt[2]++;
                } else {
                    item.removeTag(this.tags[2]);
                }
                // 清理副本
                newItem.deleted = true;
                await newItem.saveTx();
            } else {
                // 加载失败
                item.addTag(this.tags[3]);
                this.cnt[3]++;
            }
        } else {
            item.addTag(this.tags[1]);
            this.cnt[1]++;
        }
        if (is_arxiv) {
            item.setField("repository", "arXiv");
        }
    }
}


// Discard: 清空元数据
async function clearMetadata() {
    let cnt = 0;
    for (let item of ITEMS) {
        if (await getDoi(item)) {
            for (let field of FIELDS) {
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


let mu = new MetadataUpdater();
await mu.run();
// await clearMetadata();
