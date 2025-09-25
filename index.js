const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.CHANNEL_SECRET,
};

// ✅ เปลี่ยนชื่อ class ให้เข้าใจง่าย
const classNames = [
    'หยุด (STOP)',
    'จำกัดความเร็ว (Speed Limit)',
    'ทางม้าลาย (Pedestrian Crossing)',
    'ห้ามกลับรถ (U-Turn)',
    'เลี้ยว (Turn)',
    'ห้ามเลี้ยว (No Turn)',
    'โดยเฉพาะ (Especially)',
    'มีเกาะกลางถนน (Dividers)',
    'ทางเบี่ยง (Detour)'
];

const modelUrl = 'https://teachablemachine.withgoogle.com/models/kRF8-mnf8/model.json';

const app = express();
const client = new line.Client(config);
let model;

app.post('/webhook', line.middleware(config), (req, res) => {
    Promise.all(req.body.events.map(handleEvent))
        .then((result) => res.json(result))
        .catch((err) => {
            console.error(err);
            res.status(500).end();
        });
});

async function handleEvent(event) {
    if (event.type !== 'message' || event.message.type !== 'image') {
        return Promise.resolve(null);
    }

    try {
        const imageBuffer = await getImageBufferFromLine(event.message.id);

        // ✅ ปรับภาพให้เหมือนกับตอน train
        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(127.5))
            .sub(tf.scalar(1))
            .expandDims();

        const predictionResult = await model.predict(imageTensor).data();

        // ✅ หาค่าที่มีความแม่นยำมากที่สุด
        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);

        // ✅ ปรับข้อความให้ดูดีขึ้น
        let replyText = '';
        if (confidence < 40) {
            replyText =
                `🤔 ฉันไม่มั่นใจว่ารูปภาพนี้คือป้ายอะไรเลยค่ะ\n` +
                `📸 ลองถ่ายรูปใหม่ให้ชัดเจนมากขึ้น แล้วส่งมาอีกครั้งนะคะ`;
        } else {
            replyText =
                `🔍 *ผลการวิเคราะห์รูปภาพของคุณ:*\n\n` +
                `📸 ป้ายจราจร: *${bestPrediction.className}*\n` +
                `🎯 ความแม่นยำ: ${confidence}%\n\n` +
                `หากไม่ถูกต้อง คุณสามารถลองถ่ายรูปใหม่ให้ชัดขึ้นได้นะครับ 🙂`;
        }

        return client.replyMessage(event.replyToken, {
            type: 'text',
            text: replyText,
        });

    } catch (error) {
        console.error(error);
        return client.replyMessage(event.replyToken, {
            type: 'text',
            text: 'ขออภัยค่ะ เกิดข้อผิดพลาดบางอย่าง 😥',
        });
    }
}

function getImageBufferFromLine(messageId) {
    return new Promise((resolve, reject) => {
        client.getMessageContent(messageId)
            .then((stream) => {
                const chunks = [];
                stream.on('data', (chunk) => { chunks.push(chunk); });
                stream.on('error', (err) => { reject(err); });
                stream.on('end', () => { resolve(Buffer.concat(chunks)); });
            });
    });
}

async function startServer() {
    try {
        console.log('📦 กำลังโหลดโมเดล...');
        model = await tf.loadLayersModel(modelUrl);
        console.log('✅ โหลดโมเดลสำเร็จ!');
        const port = process.env.PORT || 3000;
        app.listen(port, () => {
            console.log(`🚀 Bot is ready on port ${port}`);
        });
    } catch (error) {
        console.error('❌ โหลดโมเดลล้มเหลว:', error);
    }
}

startServer();
