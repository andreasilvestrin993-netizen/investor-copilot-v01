"""
OpenAI service for GPT-4 summarization of market analysis
"""
from openai import OpenAI

def summarize_texts(texts: list[str], openai_key: str, date_str: str, mode="daily"):
    """
    Summarize market analysis texts using GPT-4.
    
    Args:
        texts: List of source texts to summarize
        openai_key: OpenAI API key
        date_str: Date string for the summary
        mode: Summary mode ("daily" or "weekly")
    
    Returns:
        Formatted summary text with sections
    """
    if not openai_key:
        return f"[{mode.upper()} {date_str}] OpenAI key missing. Paste it in the sidebar."
    
    client = OpenAI(api_key=openai_key)
    prompt = f"""
You are an investment analyst. Summarize the following sources into a single {mode} brief for {date_str}.
Output sections in this exact order with short bullet points:

1) One-paragraph market summary
2) Macro trend & expectations (up / down / sideways, key catalysts)
3) Top news (most relevant 5)
4) Spotlights (stocks, sectors, events) with 1–2 bullets each
5) Stock picks: 
   - Watch: 3–5 tickers + one-line reason 
   - Buy: 1–3 tickers + entry rationale and risk

Be concise and practical.
"""
    joined = "\n\n".join([f"Source {i+1}:\n{t[:8000]}" for i, t in enumerate(texts if texts else ["(no sources)"])])
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Be concise, actionable, neutral."},
            {"role": "user", "content": prompt + "\n\n" + joined}
        ]
    )
    return chat.choices[0].message.content.strip()
