# 🚦 RoadWatch AI - System Flow (INGRESS → INFERENCE → INSIGHTS)

This document visualizes the **end-to-end data pipeline** of RoadWatch AI, transforming raw video input into actionable road intelligence.

---

## ⚡ Quick Understanding (10 sec)

📥 **Ingress** → 🎯 **Inference** → 📊 **Insights**

- Capture road video  
- Detect hazards using AI  
- Convert into geospatial insights  
- Store + visualize in dashboard  

---

## 🔄 Full Pipeline Flowchart

```mermaid
flowchart LR

    %% =========================
    %% 🟦 STAGE 1: INGRESS
    %% =========================
    subgraph A["📥 INGRESS LAYER"]
        CAM["📷 Video Source<br/>(Dashcam / Mobile / CCTV)"]
    end

    %% =========================
    %% 🟩 STAGE 2: INFERENCE
    %% =========================
    subgraph B["🎯 INFERENCE LAYER"]
        DET["🧠 YOLOv11 Detection Engine<br/>(Real-time Object Detection)"]
        GPS["📍 GPS Tagging<br/>(Latitude / Longitude)"]
        GEO["🌍 Reverse Geocoding<br/>(Google Maps API)"]
    end

    %% =========================
    %% 🟧 STAGE 3: INSIGHTS
    %% =========================
    subgraph C["📊 INSIGHTS LAYER"]
        API["🔗 Flask API<br/>(Data Processing Layer)"]
        DB["🗄️ MongoDB<br/>(Data Storage)"]
        DASH["📊 Dashboard<br/>(Plotly Dash Visualization)"]
    end

    %% Flow Connections
    CAM -->|Frame Stream| DET
    DET -->|Detection Data| GPS
    GPS -->|Coordinates| GEO
    GEO -->|Enriched Data| API
    API -->|Async Storage| DB
    DB -->|Live Data Feed| DASH

    %% Styling
    style A fill:#e0f2fe,stroke:#0284c7,stroke-width:2px
    style B fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style C fill:#ffedd5,stroke:#ea580c,stroke-width:2px