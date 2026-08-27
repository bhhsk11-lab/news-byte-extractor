// NEWS BYTE Option 1 integration.
// Replace this with your deployed Hugging Face Space URL.
const EXTRACTOR_URL = "https://YOUR-SPACE-NAME.hf.space";

async function extractSourceArticle(articleUrl) {
    const response = await fetch(`${EXTRACTOR_URL}/extract`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            url: articleUrl,
            render: true,
            max_chars: 60000
        })
    });

    if (!response.ok) {
        throw new Error(`Extractor returned HTTP ${response.status}`);
    }

    return await response.json();
}

// Example:
//
// const article = await extractSourceArticle(newsItem.url);
//
// article.title
// article.image
// article.published
// article.author
// article.text
// article.paragraphs
// article.extraction_score
//
// IMPORTANT:
// Your NEWS BYTE briefing engine should use article.paragraphs/article.text.
// The headline alone should NOT be used as the briefing source.
