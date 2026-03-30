import requests
import json

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def generate_policy_recommendations(
    shap_dict: dict,
    cluster_id: int,
    predicted_crime: float,
    baseline_crime: float,
    feature_labels: dict,
) -> str:
    """
    Call Claude API to generate evidence-based policy recommendations.

    Args:
        shap_dict:        {feature_name: shap_value} for the current prediction
        cluster_id:       K-Means cluster (0 = highest crime, 2 = lowest)
        predicted_crime:  Model-predicted crime rate (crmrte)
        baseline_crime:   Dataset mean crime rate for comparison
        feature_labels:   Human-readable feature names

    Returns:
        Markdown-formatted policy recommendation string
    """
    # Build a readable summary of top SHAP drivers
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = "\n".join(
        f"  - {feature_labels.get(k, k)}: SHAP = {v:+.5f} "
        f"({'increases' if v > 0 else 'decreases'} crime)"
        for k, v in sorted_shap[:6]
    )

    direction = "above" if predicted_crime > baseline_crime else "below"
    pct_diff = abs((predicted_crime - baseline_crime) / baseline_crime * 100)

    cluster_descriptions = {
        0: "high-crime / low-policing cluster",
        1: "moderate-crime / average-resources cluster",
        2: "low-crime / well-resourced cluster",
    }
    cluster_desc = cluster_descriptions.get(cluster_id, "unknown cluster")

    prompt = f"""You are a criminology policy advisor analyzing North Carolina county crime data from 1981-1987. 
A data science model has produced the following findings for a county profile:

PREDICTED CRIME RATE: {predicted_crime:.5f} (crmrte)
BASELINE (dataset mean): {baseline_crime:.5f}
STATUS: {pct_diff:.1f}% {direction} the state average
COUNTY CLUSTER: {cluster_desc} (Cluster {cluster_id})

TOP SHAP FEATURE DRIVERS (what is most influencing this prediction):
{top_drivers}

Based on these specific findings, provide:
1. **Situation Summary** (2-3 sentences interpreting what the numbers mean in plain English)
2. **Top 3 Policy Recommendations** — each must be directly tied to the SHAP drivers above, specific and actionable, with an expected mechanism of impact
3. **Risk Factors to Monitor** — 2 warning signs based on the data profile
4. **One Data Limitation** — an honest note about what this 1981-87 dataset cannot tell us

Keep the tone professional but accessible. Use markdown formatting. Be specific — no generic crime-reduction platitudes."""

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            CLAUDE_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not reach Claude API: {str(e)}\n\nPlease check your API key configuration."
    except (KeyError, IndexError) as e:
        return f"⚠️ Unexpected API response format: {str(e)}"


def generate_cluster_narrative(cluster_profiles: dict) -> str:
    """
    Generate a brief LLM narrative describing the three cluster profiles.
    Used in the Analytics Dashboard tab.
    """
    profiles_text = ""
    for cid, row in cluster_profiles.iterrows():
        profiles_text += f"\nCluster {cid}:\n"
        for col, val in row.items():
            profiles_text += f"  {col}: {val:.4f}\n"

    prompt = f"""You are a criminology researcher. Below are mean feature values for 3 county clusters 
derived from K-Means clustering of North Carolina county crime data (1981-1987):

{profiles_text}

In 3 short paragraphs (one per cluster), describe:
- What type of county this cluster represents
- Its most distinctive characteristics
- One concrete policy implication

Be concise and analytical. Use plain language."""

    payload = {
        "model": MODEL,
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            CLAUDE_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"⚠️ Could not generate cluster narrative: {str(e)}"
