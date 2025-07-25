# 接口要求

- 请求协议：ws[s]（为提高安全性，强烈推荐wss）
- 请求地址：方言识别地址：ws[s]: //iat.cn-huabei-1.xf-yun.com/v1
- 接口鉴权：签名机制，详情请参照下方接口鉴权
- 字符编码：UTF-8
- 响应格式：统一采用JSON格式
- 开发语言：任意，只要可以向讯飞云服务发起HTTP请求的均可
- 音频属性：采样率16k或8K、位长16bit、单声道
- 音频格式：pcm
- 音频长度：最长60s

# 接口鉴权

通过在请求地址后面加上鉴权相关参数的方式。示例url：

```
wss://iat.cn-huabei-1.xf-yun.com/v1?authorization=YXBpX2tleT0iYTFkNjc0NmRkMWJiMTlmMTlkNTkyMDVhZDAwMDc0NjQiLCBhbGdvcml0aG09ImhtYWMtc2hhMjU2IiwgaGVhZGVycz0iaG9zdCBkYXRlIHJlcXVlc3QtbGluZSIsIHNpZ25hdHVyZT0iRDZFQzFCRHk5Wjl2Y1RqdE55aW0wenNFZi9LQUxIQmg1TlNxYVcwMUNJbz0i&amp;date=Mon%2C+30+Dec+2024+07%3A40%3A12+GMT&amp;host=iat.cn-huabei-1.xf-yun.com
```

鉴权方法

|参数|类型|必须|说明|示例|
|--|--|--|--|--|
|host|string|是|请求主机|iat.xf-yun.com|
|date|string|是|当前时间戳，RFC1123格式|Tue, 14 May 2024 08:46:48 GMT|
|authorization|string|是|使用base64编码的签名相关信息(签名基于hmac-sha256计算)|参考下方authorization参数生成规则|

date参数生成规则：
date必须是UTC+0或GMT时区，RFC1123格式(Tue, 14 May 2024 08:46:48 GMT)。 服务端会对Date进行时钟偏移检查，最大允许300秒的偏差，超出偏差的请求都将被拒绝。

authorization参数生成规则：
1）获取接口密钥APIKey 和 APISecret。 在讯飞开放平台控制台，创建WebAPI平台应用并添加语音听写（流式版）服务后即可查看，均为32位字符串。
2）参数authorization base64编码前（authorization_origin）的格式如下。
```
api_key="$api_key",algorithm="hmac-sha256",headers="host date request-line",signature="$signature"
```
其中 api_key 是在控制台获取的APIKey，algorithm 是加密算法（仅支持hmac-sha256），headers 是参与签名的参数（见下方注释）。 signature 是使用加密算法对参与签名的参数签名后并使用base64编码的字符串，详见下方。
注：headers是参与签名的参数，请注意是固定的参数名（"host date request-line"），而非这些参数的值。
3）signature的原始字段(signature_origin)规则如下。
signature原始字段由 host，date，request-line三个参数按照格式拼接成， 拼接的格式为(\n为换行符,’:’后面有一个空格)：
```
host: $host\ndate: $date\n$request-line
```
假设
```
请求url = wss://iat.cn-huabei-1.xf-yun.com/v1
date = Tue, 14 May 2024 08:46:48 GMT
```
那么 signature原始字段(signature_origin)则为：
```
host: iat.cn-huabei-1.xf-yun.com
date: Tue, 14 May 2024 08:46:48 GMT
GET /v1 HTTP/1.1
```
4）使用hmac-sha256算法结合apiSecret对signature_origin签名，获得签名后的摘要signature_sha。
```
signature_sha=hmac-sha256(signature_origin,$apiSecret)
```
其中 apiSecret 是在控制台获取的APISecret
5）使用base64编码对signature_sha进行编码获得最终的signature。
```
signature=base64(signature_sha)
```
假设
```
APISecret = secretxxxxxxxx2df7900c09xxxxxxxx	
date = Tue, 14 May 2024 08:46:48 GMT
```
则signature为
```
signature=S66FeqTJlvkd+KfJg+a73BAabocwRg2scKflONI8o80=
```
6）根据以上信息拼接authorization base64编码前（authorization_origin）的字符串，示例如下。
```
api_key="keyxxxxxxxx8ee279348519exxxxxxxx", algorithm="hmac-sha256", headers="host date request-line", signature="S66FeqTJlvkd+KfJg+a73BAabocwRg2scKflONI8o80="
```
注： headers是参与签名的参数，请注意是固定的参数名（"host date request-line"），而非这些参数的值。
7）最后再对authorization_origin进行base64编码获得最终的authorization参数。
```
authorization = base64(authorization_origin)
示例：
wss://iat.cn-huabei-1.xf-yun.com/v1?authorization=YXBpX2tleT0iYTFkNjc0NmRkMWJiMTlmMTlkNTkyMDVhZDAwMDc0NjQiLCBhbGdvcml0aG09ImhtYWMtc2hhMjU2IiwgaGVhZGVycz0iaG9zdCBkYXRlIHJlcXVlc3QtbGluZSIsIHNpZ25hdHVyZT0iRDZFQzFCRHk5Wjl2Y1RqdE55aW0wenNFZi9LQUxIQmg1TlNxYVcwMUNJbz0i&amp;date=Mon%2C+30+Dec+2024+07%3A40%3A12+GMT&amp;host=iat.cn-huabei-1.xf-yun.com
```

# 数据传输接收

握手成功后客户端和服务端会建立Websocket连接，客户端通过Websocket连接可以同时上传和接收数据。 当服务端有识别结果时，会通过Websocket连接推送识别结果到客户端。

发送数据时，如果间隔时间太短，可能会导致引擎识别有误。 建议每次发送音频间隔40ms，每次发送音频字节数（即java示例demo中的frameSize）为一帧音频大小的整数倍。

```java
//连接成功，开始发送数据
int frameSize = 1280; //每一帧音频大小的整数倍，请注意不同音频格式一帧大小字节数不同，可参考下方建议
int intervel = 40;
int status = 0;  // 音频的状态
try (FileInputStream fs = new FileInputStream(file)) {
    byte[] buffer = new byte[frameSize];
    // 发送音频
```
我们建议：未压缩的PCM格式，每次发送音频间隔40ms，每次发送音频字节数1280B；

# 请求协议示例

```json
{
    "header": {
        "app_id": "123456",
    },
    "parameter": {
        "iat": {
            "language": "zh_cn",
            "accent": "mulacc",
            "domain": "slm",
            "eos": 1800,
            "dwa": "wpgs",
            "ptt": 1,
            "nunum": 1,
            "ltc": 0,
            "result": {
                "encoding": "utf8",
                "compress": "raw",
                "format": "json"
            }
        }
    },
    "payload": {
        "audio": {
            "encoding": "raw",
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "status": 0,
            "seq": 0,
            "audio": "AAAAAP...",
        }
    }
}
```

|字段|含义|类型|说明|
|--|--|--|--|
|header|协议头部|Object|协议头部，用于描述平台特性的参数，详见 5.2 平台参数。|
|parameter|能力参数|Object|AI 特性参数，用于控制 AI 引擎特性的开关。|
|iat|服务别名|Object||
|result|响应数据控制|Object|数据格式预期，用于描述返回结果的编码等相关约束，不同的数据类型，约束维度亦不相同，此 object 与响应结果存在对应关系。|
|payload|输入数据段|Object|数据段，携带请求的数据。|
|audio|输入数据|Object|输入数据，详见 5.2 请求数据。|

# 请求参数

平台参数

|字段|含义|类型|是否必传|
|--|--|--|--|
|app_id|在平台申请的app id信息|string|是|
|res_id|应用级热词，用于提高相关词语识别权重（可直接在控制台设置，并上传res_id）|string|否|
|status|请求状态，可选值为：0-开始、1-继续、2-结束|int|是|

功能参数

|功能标识|功能描述|数据类型|取值范围|必填|默认值|
|--|--|--|--|--|--|
|language|语种设置|String|仅支持：zh_cn|是|zh_cn|
|accent|方言设置|String|仅支持：mulacc|支持202种方言免切换，具体可查看支持方言明细|是|mulacc|
|domain|应用领域设置|String|仅支持：slm|是|slm|
|eos|尾静音截断：引擎判定结束的时间，连续检测给定时间长度的音频，均为静音，则引擎停止识别|int|最小值:600, 最大值:10000|否|1800|
|nbest|句子多候选：通过参数控制输出的n个最优的结果，而不是1个|int|最小值:0, 最大值:5|否|0|
|wbest|词级多候选：通过控制输出槽内的n个结果，而不是1个|int|最小值:0, 最大值:5|否|0|
|vinfo|句子级别帧对齐:给出一次会话中，子句的vad边界信息|int|0:不返回vad信息, 1:返回vad信息|否|0|
|dwa|流式识别PGS：流式识别功能，打开后，会话过程中实时给出语音识别的结果，而不是子句结束时才给结果|string|最小长度:0, 最大长度:10|否||
|ptt|标点预测：在语音识别结果中增加标点符号|int|0:关闭, 1:开启|否|1|
|smth|顺滑功能：将语音识别结果中的顺滑词（语气词、叠词）进行标记，业务侧通过标记过滤语气词最终展现识别结果|int|0:关闭, 1:开启|否|0|
|nunum|数字规整：将语音识别结果中的原始文字串转为相应的阿拉伯数字或者符号|int|0:关闭, 1:开启|否|1|
|opt|是否输出属性|int|0:json格式输出，不带属性, 1:文本格式输出，不带属性, 2:json格式输出，带文字属性"wp":"n"和标点符号属性"wp":"p"|否||
|dhw|会话热词，支持utf-8和gb2312；取值样例：“dhw=gb2312;你好大家”（对应gb2312编码）；“dhw=utf-8;你好|大家”（对应utf-8编码）|string|最小长度:0, 最大长度:1024|否||
|rlang|返回字体指定zh-cn/zh-hk/zh-mo/zh-tw，服务默认是简体字|string|最小长度:0, 最大长度:100|否||
|ltc|是否进行中英文筛选|int|1:不进行筛选, 2:只出中文, 3:只出英文|否||

响应数据参数

|字段|含义 |数据类型|取值范围 |默认值|说明 |必填|
|--|--|--|--|--|--|--|
|encoding|文本编码|string|utf8, gb2312 |utf8|取值范围可枚举|否|
|compress|文本压缩格式|string|raw, gzip |raw|取值范围可枚举|否|
|format|文本格式|string|plain, json, xml|json|取值范围可枚举|否|

请求数据
audio（默认请求）

|字段|含义|数据类型|取值范围|默认值|说明|必填|
|--|--|--|--|--|--|--|
|encoding|音频编码|string|lame（即mp3格式）, speex, opus, opus-wb, speex-wb|speex-wb|取值范围可枚举|否|
|sample_rate|采样率|int|16000, 8000|16000|音频采样率，可枚举|否|
|channels|声道数|int|1, 2|1|声道数，可枚举|否|
|bit_depth|位深|int|16, 8|16|单位bit，可枚举|否|
|status|数据状态|int|0:开始, 1:继续, 2:结束|0|取值范围为：0（开始）、1（继续）、2（结束）|否|
|seq|数据序号|int|最小值:0, 最大值:9999999|0|标明数据为第几块|否|
|audio|音频数据|string|最小尺寸:0B, 最大尺寸:10485760B||音频大小：0-10M|是|
|frame_size|帧大小|int|最小值:0, 最大值:1024|0|帧大小，默认0|否|

# 响应协议示例

```json
{
    "header": {
        "code": 0,
        "message": "success",
        "sid": "ase000e065f@dx193f81b97fb750c882",
        "status": 2
    },
    "payload": {
        "result": {
            "encoding": "utf8",
            "compress": "raw",
            "format": "json",
            "status": 0,
            "seq": 0,
            "text": "eyJzbiI6MSwibHMiOnRydWUsImJnIjowLCJlZCI6MCwid3MiOlt7ImJnIjowLCJjdyI6W3……"
        }
    }
}
```

- header
  - code: 返回码，0表示成功，其它表示异常
  - message: 错误描述
- payload

|字段|含义|数据类型|取值范围|默认值|说明|必填|
|--|--|--|--|--|--|--|
|encoding|文本编码|string|utf8, gb2312|utf8|取值范围可枚举|否|
|compress|文本压缩格式|string|raw, gzip|raw|取值范围可枚举|否|
|format|文本格式|string|plain, json, xml|json|取值范围可枚举|否|
|status|数据状态|int|0:开始, 1:继续, 2:结束|0|取值范围为：0（开始）、1（继续）、2（结束）|否|
|seq|数据序号|int|最小值:0, 最大值:9999999|0||是|
|text|文本数据|string|最小长度:0, 最大长度:1000000|||是|

result.text 示例

```json
{
    "bg": 0,
    "ed": 0,
    "ls": false,
    "sn": 15,
    "pgs": rpl,
    "rst": pgs,
    "rg": [
        1,14
    ],
    "ws": [
        {
            "wb": null,
            "wc": null,
            "we": null,
            "wp": null,
            "cw": [
                {
                    "sc": 0,
                    "ph": null,
                    "w": 科大讯飞
                }
            ]
        }
    ]
}
```

|字段|含义|数据类型|取值范围|默认值|说明|
|--|--|--|--|--|--|
|bg|float|--|140|本次识别结果的语音开始端点，以ms为单位|
|ed|float|--|2280|本次识别结果的语音结束端点，以ms为单位|
|ls|boolean|true:false:|false|本次结果是否为最后一块结果|
|sn|float|--|1|本次识别结果在总体识别结果中的序号|
|pgs|string|--|rpl|流式识别场景下，本次识别结果操作方式，rpl 为替换前一次识别结果，apd为替换前一次识别结果|
|rst|string|--|rlt|流式识别场景下，本地识别结果的类型，rlt为子句最终结果，pgs 为子句过程的流式结果|
|rg|array|--|[]|流式识别场景下，结果标识字段，字段为2维数组，第一个值为 sn 的值，第二个为替换子句的终止sn号|
|ws|array|--|[]|本次识别结果的内容，是一个多维数组，每个值表示一个槽|

