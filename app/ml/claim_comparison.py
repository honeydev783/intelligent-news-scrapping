import numpy as np
from ml.embeddings import embed
from ml.llm import call_llm
from db.models import Claim

def cosine_similarity(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")

    if a.shape != b.shape:
        raise ValueError(f"Embedding shape mismatch: {a.shape} vs {b.shape}")

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)



def llm_contradiction_check(text1: str, text2: str) -> str:
    prompt = """
    Determine the relationship between two claims.

    Return JSON with:
    - relationship: supporting | contradicting | unrelated
    """

    try:
        result = call_llm(
            prompt,
            f"CLAIM A: {text1}\nCLAIM B: {text2}"
        )
    except TypeError as e:
        print("LLM CALL SIGNATURE ERROR:", e)
        return "unrelated"

    if not isinstance(result, dict):
        print("LLM RAW RESULT:", result)
        return "unrelated"

    return result.get("relationship", "unrelated")



def compare_claims(
    claims: list[Claim],
    semantic_threshold: float = 0.75,
):
    if len(claims) < 2:
        return []

    embeddings = {}

    for c in claims:
        raw = embed(c.claim_text)

        # Normalize embedding shape
        if isinstance(raw, list) and isinstance(raw[0], list):
            raw = raw[0]

        vec = np.array(raw, dtype="float32")

        if vec.ndim != 1:
            raise ValueError(f"Invalid embedding shape: {vec.shape}")

        embeddings[c.id] = vec
    results = set()
    try: 
        for i, c1 in enumerate(claims):
            for j, c2 in enumerate(claims):
                if i >= j:
                    continue

                sim = cosine_similarity(
                    embeddings[c1.id],
                    embeddings[c2.id],
                )

                if sim < semantic_threshold:
                    continue

                relationship = llm_contradiction_check(
                    c1.claim_text,
                    c2.claim_text
                )
                print("LLM relationship raw:", relationship, type(relationship))
                if relationship in ("supporting", "contradicting"):
                    results.add((c1.id, relationship))
                    results.add((c2.id, relationship))
    except Exception as e:
        print("comparison exception===>", str(e))
        
    return list(results)

