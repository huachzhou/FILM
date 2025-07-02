# Self-Monitoring Large Language Models for Click-Through Rate Prediction

The sample data from Amazon Movies and example fine-tuning code for FILM and Base LLMs. Complete code and data will be released upon paper acceptance.

## Performance

On this sample data:
- **FILM performance**: 
  - AUC: 0.8670 
  - Logloss: 0.2796 
  - (3 pre-training and 4 fine-tuning epochs)
- **Base LLMs**: 
  - AUC: 0.8485 
  - Logloss: 0.2945 
  - (4 fine-tuning epochs)

## Setup

The python packages are in the `requirement.txt` and you could install necessary ones.

## Running the Code

To run the code:

```bash
./shell/instruct_film.sh GPU_id

## 📮 Contact
For any questions or feedback, please contact:
- Huachi Zhou - huachi.zhou@connect.polyu.hk
