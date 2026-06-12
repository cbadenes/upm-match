<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="app/static/horizontal-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="app/static/stacked.svg">
  <img alt="UPM Match" src="app/static/stacked.svg" width="320">
</picture>

<br/>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-white?logo=ollama&logoColor=black)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PIE IE26.6102](https://img.shields.io/badge/PIE-IE26.6102-0072CE)](https://innovacioneducativa.upm.es/proyectos-ie/informacion?anyo=2025-2026&id=2018)

</div>

---

UPM Match is an intelligent recommendation system designed to help students discover suitable academic supervisors and to foster research collaborations among faculty members at Universidad Politécnica de Madrid (UPM).

The system analyzes publicly available data from the UPM Scientific Portal to build semantic profiles of researchers based on their publications, projects, research topics, and past supervision activity.

Using natural language processing and semantic similarity techniques, UPM Match recommends:

- Potential supervisors for Bachelor's, Master's, or PhD theses.
- Potential collaborations between researchers with related research interests.

---

## Motivation

Choosing a thesis supervisor is often difficult for students due to limited knowledge of the research landscape within their institution.

UPM Match aims to provide a **data-driven and transparent approach** to this process by leveraging institutional research data to suggest supervisors aligned with the student's interests.

Additionally, the system promotes **interdisciplinary collaboration** among faculty by identifying thematic affinities between researchers.

---

## Main Features

- Semantic representation of faculty research profiles
- Tutor recommendation for:
  - Bachelor's theses (TFG)
  - Master's theses (TFM)
  - PhD supervision
- Detection of potential collaborations between researchers
- Exploration of research topics and academic expertise
- Prototype web interface for students and faculty

---

## How It Works

The system is based on several components:

1. **Data extraction**
   - Collection of public data from the UPM Scientific Portal.

2. **Profile construction**
   - Aggregation of publications, projects, and academic activities into researcher profiles.

3. **Semantic representation**
   - Use of NLP techniques (embeddings, clustering, similarity analysis) to model research topics.

4. **Recommendation engine**
   - Matching student interests with relevant supervisors.
   - Detecting potential research collaborations through thematic similarity.

---

## Project Context

<table><tr>
<td>

UPM Match is developed as part of the **Educational Innovation Project (IE26.6102)** funded by Universidad Politécnica de Madrid.

The project explores how **Artificial Intelligence can support academic decision-making and collaboration within universities.**

</td>
<td align="center" valign="middle" width="180">

<a href="https://www.etsisi.upm.es"><img src="https://www.upm.es/gsfs/SFS11384" alt="ETSISI" height="40"/></a>&nbsp;&nbsp;<a href="https://www.upm.es"><img src="https://www.upm.es/gsfs/SFS46900" alt="UPM" height="40"/></a>

</td>
</tr></table>
