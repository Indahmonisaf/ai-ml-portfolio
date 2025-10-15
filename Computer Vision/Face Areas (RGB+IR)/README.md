
* **Face Areas (RGB+IR)** — Multimodal detection of forehead/nose/cheeks with RGB–IR fusion; robust in low light.  <sb>
  **Goal:** detect nostril and nose areas to estimate respiration rate.
  **Dataset:** Public dataset + AVIL dataset; Train **3,391** images, Val **338** images. 
  **Method:**  
  - **Nostril detection:** leverage infrared to capture airflow-induced thermal changes. 
  - **Nose-area detection:** track global nose motion to capture breathing signal. 
