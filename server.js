const express = require('express');
const fs = require('fs');
const fetch = require('node-fetch');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const DATA_FILE = './visits.json';

function readData() {
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, 'utf-8'));
  } catch (e) {
    return [];
  }
}

function writeData(data) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
}

app.post('/api/visit', async (req, res) => {
  const { ip } = req.body;
  let location = { country: 'Unknown' };
  if (ip) {
    try {
      const resp = await fetch(`https://ipapi.co/${ip}/json/`);
      if (resp.ok) {
        location = await resp.json();
      }
    } catch (e) {
      console.error('Failed to fetch location', e);
    }
  }
  const visits = readData();
  visits.push({
    time: new Date().toISOString(),
    ip,
    location,
    page: req.body.page || ''
  });
  writeData(visits);
  res.json({ ok: true });
});

app.get('/api/visits', (req, res) => {
  res.json(readData());
});

app.get('/api/stats', (req, res) => {
  const visits = readData();
  const perDay = {};
  const perCountry = {};
  visits.forEach(v => {
    const day = v.time.substring(0, 10);
    perDay[day] = (perDay[day] || 0) + 1;
    const country = v.location && v.location.country_name ? v.location.country_name : 'Unknown';
    perCountry[country] = (perCountry[country] || 0) + 1;
  });
  res.json({ perDay, perCountry });
});

app.listen(PORT, () => {
  console.log(`Analytics server running on port ${PORT}`);
});