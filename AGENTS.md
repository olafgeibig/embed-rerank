This project is an embedding and re-ranking service that can run locally. I am as I'm on a Mac I want to use it with MLX. I'm particularly interested in the re-ranking, but I want to use the embeddings as well. 

- I'm running this service with `uv run embed-rerank`
- The config is in .env

## Reranking test requests
```
# correct answer is 3,6,2

curl http://localhost:9010/api/v1/rerank/ -H "Content-Type: application/json" -d '{
      "query": "Organic skincare products for sensitive skin",
      "documents": [
        "Eco-friendly kitchenware for modern homes",
        "Biodegradable cleaning supplies for eco-conscious consumers",
        "Organic cotton baby clothes for sensitive skin",
        "Natural organic skincare range for sensitive skin",
        "Tech gadgets for smart homes: 2024 edition",
        "Sustainable gardening tools and compost solutions",
        "Sensitive skin-friendly facial cleansers and toners",
        "Organic food wraps and storage solutions",
        "All-natural pet food for dogs with allergies",
        "Yoga mats made from recycled materials"
      ],
      "top_n": 3
    }'
```

Test with bigger documents:
```
# expected answer: 0,3,9
curl http://localhost:=9010/api/v1/rerank/ -H "Content-Type: application/json" -d '{
  "query": "The physiological impact of high-altitude training on long-distance athletic endurance and oxygen transport",
  "top_n": 3,
  "documents": [
    "High-altitude training, conducted above 2,400 meters, is a staple for elite athletes. The primary physiological adaptation is the surge in erythropoietin (EPO) production, a hormone from the kidneys that triggers red blood cell creation. By increasing hemoglobin mass, athletes boost the oxygen-carrying capacity of their blood, aiding performance at sea level. However, the lower oxygen pressure can reduce training intensity, potentially leading to muscle detraining. To counter this, many use the Live-High, Train-Low (LHTL) model, resting at altitude for hematological gains but descending for high-intensity workouts. Studies show a four-week stay is necessary for measurable VO2 max improvements. Without careful management, the stress of hypoxia can lead to overtraining or sleep disturbances. This document explores the complex relationship between hemoglobin mass, oxygen saturation, and the metabolic cost of maintaining power output in rarefied air conditions over long durations. (Added padding for length: 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789)",
    "The 1968 Olympic Games in Mexico City (2,240m) acted as a global laboratory for altitude physiology. While sprinters thrived due to low air resistance, endurance athletes suffered. This event catalyzed research into how the body manages oxygen transport under stress. Post-1968, centers in Colorado Springs and St. Moritz became vital for training. The focus shifted to understanding VO2 max and the kidneys response to hypoxia. This retrospective looks at the data from distance runners who experienced significant drops in aerobic capacity during the games. It analyzes how blood pH shifts as the body breathes harder to expel CO2, a process known as respiratory alkalosis. This historical data is essential for modern coaches planning periodization for events at varying elevations. (Added padding for length: 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789)",
    "High-altitude cooking requires adjustments in fluid dynamics. As pressure drops, the boiling point of water decreases, meaning pasta and grains take longer to soften. Bakers must reduce leavening agents because air bubbles expand faster in thin air, which can cause cakes to collapse. This document covers the chemistry of heat transfer and molecular structure at 3,000 meters, specifically for industrial kitchens in the Andes. It explains why pressure cookers are essential for food safety and texture when atmospheric pressure is 30 percent lower than at sea level. It does not cover athletic performance or physiological oxygen transport, focusing purely on culinary thermodynamics and the physical properties of steam. (Added padding for length: 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789)",
    "Modern mountain photography in the Himalayas involves managing high UV levels and extreme light contrast. Thin air filters less light, leading to blue hazes in images. Photographers must use polarizers and UV filters to protect sensors and maintain color accuracy. Battery life is also a concern, as extreme cold at high altitudes drains power rapidly. This guide provides technical settings for landscape photography at 5,000 meters, including how to manage the high dynamic range of bright snow against dark rock shadows. It discusses the physical endurance of the photographer but does not provide scientific data on oxygen transport or red blood cell mass. (Added padding for length: 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789)"
  ]
}'
```
