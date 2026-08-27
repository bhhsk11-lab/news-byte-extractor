const EXTRACTOR_URL = "https://YOUR-RENDER-SERVICE.onrender.com";

async function extractSourceArticle(articleUrl) {
  const response = await fetch(`${EXTRACTOR_URL}/extract`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({url: articleUrl, render: false})
  });
  if (!response.ok) throw new Error(`Extractor HTTP ${response.status}`);
  return await response.json();
}
