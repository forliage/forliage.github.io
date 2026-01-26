(async function() {
    try {
      const ipRes = await fetch('https://api.ipify.org?format=json');
      const ipData = await ipRes.json();
      await fetch('/api/visit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip: ipData.ip, page: location.pathname })
      });
    } catch (e) {
      console.error('Failed to record visit', e);
    }
  })();