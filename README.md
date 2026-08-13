# belge-rag

Yerel makinede çalışan, PDF tabanlı bir soru-cevap (RAG) servisi. Bir PDF belgesi
sayfa sayfa metne çevrilir, parçalara bölünür, `bge-m3` ile embed'lenip
PostgreSQL/pgvector'a yazılır. `/ask` endpoint'ine gelen soru aynı modelle
embed'lenir, cosine benzerliğine göre en yakın parçalar getirilir ve benzerlik
eşiğini geçen sorular için `qwen2.5:7b` bağlam üzerinden Türkçe cevap üretir.
Eşiği geçemeyen sorulara LLM'e hiç gidilmeden "belgelerde yok" cevabı döner.
Tüm bileşenler (Ollama, Postgres) lokalde çalışır; dışarıya veri çıkmaz.

## Mimari

```mermaid
flowchart LR
    subgraph Ingest ["Ingest (python -m ingest.ingest dosya.pdf)"]
        A[PDF] --> B[pypdf<br>sayfa sayfa metin]
        B --> C[recursive chunking<br>1500 karakter / 200 overlap]
        C --> D[bge-m3<br>Ollama /api/embed]
        D --> E[(pgvector<br>chunks tablosu)]
    end
    subgraph Sorgu ["Sorgu (POST /ask)"]
        F[soru] --> G[bge-m3 embed]
        G --> H[cosine top-k<br>pgvector '&lt;=&gt;']
        H --> I{"benzerlik ≥ SIM_THRESHOLD?"}
        I -- hayır --> J["'Bu bilgi yüklü belgelerde<br>bulunmuyor.' (LLM çağrılmaz)"]
        I -- evet --> K[qwen2.5:7b<br>Ollama /v1/chat/completions<br>OpenAI-uyumlu]
        K --> L[cevap + kaynaklar]
    end
    E -.-> H
```

Chunking ayırıcı hiyerarşisiyle (paragraf → cümle → satır → kelime) çalışır;
kelime ortasından kesmez, ardışık parçalar arasında ~200 karakter örtüşme
bırakır. Aynı dosya tekrar ingest edilirse eski kayıtlar tek transaction
içinde silinip yenileri yazılır.

## Kurulum

Gereksinimler: Docker, [Ollama](https://ollama.com), Python 3.12+.

```bash
# 1. Postgres + pgvector (host portu 5434)
docker compose up -d

# 2. Modeller
ollama pull bge-m3
ollama pull qwen2.5:7b

# 3. Python ortamı
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Belge ingest
python -m ingest.ingest yonetmelik.pdf

# 5. API
uvicorn app.main:app --reload
```

Yapılandırma `.env` üzerinden (`app/config.py` varsayılanları):

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5434/belge_rag` | docker-compose ile uyumlu |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `EMBED_MODEL` | `bge-m3` | 1024 boyut; şemadaki `vector(1024)` ile eşleşmeli |
| `CHAT_MODEL` | `qwen2.5:7b` | |
| `SIM_THRESHOLD` | `0.5` | cosine similarity alt sınırı (bkz. Eşik kalibrasyonu) |

## Örnek soru-cevap

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Resmî yazışmalarda hangi yazı tipi ve punto kullanılır?"}'
```

**1. Belgede olan soru**

> **Soru:** Resmî yazışmalarda hangi yazı tipi ve punto kullanılır?
>
> **Cevap:** Belgelerde "Times New Roman" veya "Arial" yazı tipi normal yazı
> stilinde kullanılır. Harf büyüklüğü Times New Roman için 12, Arial için 11
> puntodur; gerekli hâllerde yazı alanında 9 puntoya, iletişim bilgileri
> kısmında 8 puntoya kadar düşürülebilir. Kaynak: yonetmelik.pdf, s.3.

**2. Belgede olan başka bir soru**

> **Soru:** Belgeler hangi kâğıt boyutuna göre hazırlanır?
>
> **Cevap:** Belgeler A4 (210x297 mm) boyutu dikkate alınarak hazırlanır;
> belge ekleri farklı form, format veya ebatlarda olabilir.
> Kaynak: yonetmelik.pdf, s.3.

**3. Kapsam dışı soru (eşik devreye girer)**

> **Soru:** Türkiye'nin en yüksek dağı hangisidir?
>
> **Cevap:** Bu bilgi yüklü belgelerde bulunmuyor. (`sources: []` — soru
> eşiği geçemediği için LLM hiç çağrılmaz.)

Yanıt formatı: `{"answer": "...", "sources": [{"file": "yonetmelik.pdf", "page": 3, "similarity": 0.71}]}`.
Sağlık kontrolü için `GET /health` mevcut.

## Eşik kalibrasyonu

`SIM_THRESHOLD=0.5` ölçümle seçildi. Yüklü belge üzerinde denenen alakalı
sorular en iyi chunk için 0.63–0.73 bandında benzerlik döndürdü; alakasız
sorular (belgeyle ilgisi olmayan genel bilgi soruları) ≤0.41'de kaldı. İki
küme arasında net bir boşluk olduğu için eşik boşluğun ortasına, 0.5'e kondu.

Eşiğin retrieval katmanında olmasının nedeni: alakasız bir soru bağlamla
birlikte LLM'e gönderildiğinde, system prompt "yalnızca bağlamdan cevap ver"
dediği hâlde model zaman zaman kendi bilgisinden yanlış/uydurma cevap üretti.
Prompt'a güvenmek olasılıksal bir savunma; benzerlik eşiği ise deterministik.
Bu yüzden kapsam dışı sorular LLM'e hiç ulaşmadan retrieval katmanında
reddediliyor — bu hem hallucination riskini sıfırlıyor hem de gereksiz LLM
çağrısını kesiyor.

## Bilinçli kapsam dışı ve production yolu

Bu proje tek kullanıcılı, tek belgeli bir prototip. Aşağıdakiler bilinçli
olarak kapsam dışı bırakıldı; production'a giden yol sırasıyla şunlardan geçer:

- **Yetki filtreli retrieval:** Şu an tüm chunk'lar herkese açık. Çok
  kullanıcılı senaryoda satır bazlı erişim (Postgres RLS) veya chunk
  metadata'sı üzerinden yetki filtresi retrieval sorgusuna eklenmeli.
- **Hybrid search:** Salt dense retrieval terim eşleşmesinde (madde numarası,
  özel ad, kısaltma) zayıf kalabilir. BM25 + dense sonuçlarının RRF ile
  birleştirilmesi ilk adım.
- **Reranking:** Top-k sonuçların bir cross-encoder ile yeniden sıralanması,
  özellikle k büyüdükçe bağlam kalitesini belirgin artırır.
- **HNSW index:** Şemada bilinçli olarak ANN index yok; bu veri boyutunda
  exact scan yeterli. Chunk sayısı yüz binleri bulduğunda pgvector HNSW
  index'i eklenmelidir (recall/hız dengesi ölçülerek).
- **Değerlendirme seti:** Eşik ve retrieval kalitesi şu an elle denenmiş az
  sayıda soruya dayanıyor. Soru-cevap-kaynak üçlülerinden oluşan bir
  değerlendirme seti kurulup recall@k ve cevap doğruluğu düzenli ölçülmeli.
- **vLLM'e geçiş:** Ollama tek kullanıcı için pratik; eşzamanlı istek altında
  throughput sınırlı. Production'da OpenAI-uyumlu endpoint korunarak
  qwen2.5:7b vLLM üzerinde servis edilmeli (continuous batching, daha yüksek
  eşzamanlılık).
