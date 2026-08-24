# Monitoramento Inteligente de Condições de Motores Elétricos

Prova de conceito para classificação de condições de motores elétricos a partir de sinais vibroacústicos. O pipeline cobre análise exploratória, engenharia de features, validação sem mistura de janelas da mesma gravação, otimização de hiperparâmetros e comparação entre Random Forest e SVM com kernel RBF.

## 1. Objetivo

O projeto implementa o Módulo 1 do desafio de IA Industrial, classificar quatro condições de um motor de indução trifásico a partir de sinais de vibração e áudio:

- `H_H`: Healthy;
- `R_U`: Unbalance;
- `R_M`: Misalignment;
- `B_R`: Bowed rotor.

## 2. Dataset

Foi utilizado um subset curado do University of Ottawa Electric Motor Dataset: Vibration and Acoustic Faults under Constant and Variable Speed Conditions (UOEMD-VAFCVS).

O subset contém:

- 16 arquivos CSV, com 4 arquivos por classe;
- 420.000 amostras por arquivo;
- 10 s de aquisição por arquivo;
- frequência de amostragem de 42 kHz;
- duas velocidades nominais do VFD: 15 Hz e 45 Hz;
- duas condições de carga: unloaded e loaded;
- três acelerômetros, um microfone e um sensor de temperatura.

No Módulo 1 foram utilizados apenas os quatro canais vibroacústicos. A temperatura foi mantida fora do classificador porque apresenta dinâmica significativamente mais lenta e é tratada no enunciado como variável-alvo do Módulo 2.

Fonte do dataset: Sehri, M.; Dumond, P.; Bouchard, M. *University of Ottawa constant and variable speed electric motor vibration and acoustic fault signature dataset*. Data in Brief, 53, 109327, 2024. Dataset: DOI `10.17632/msxs4vj48g`, licença CC BY 4.0.

## 3. Estrutura do projeto

```text
.
├── main.ipynb
├── eda_functionc.py
├── features_extraction.py
├── classification.py
├── requirements.txt
└── README.md
```

- `main.ipynb`: execução e documentação do pipeline completo.
- `eda_functionc.py`: leitura dos sinais, reconstrução dos metadados, verificações de qualidade e visualizações da EDA.
- `features_extraction.py`: janelamento, extração e análise de features, redução de redundância e geração dos splits agrupados.
- `classification.py`: modelos, busca de hiperparâmetros, nested cross-validation, métricas e visualizações de desempenho.

## 4. Stack e justificativas

A implementação foi desenvolvida em Python com uma stack orientada a dados tabulares e processamento de sinais:

- NumPy: operações vetorizadas, FFT e processamento numérico;
- Pandas: leitura dos CSVs e organização de metadados, features e resultados;
- SciPy: estatísticas descritivas e operações auxiliares de análise de sinais;
- Matplotlib: EDA e visualização dos resultados;
- Scikit-learn: seleção de features, validação agrupada, pré-processamento, Random Forest, SVM e métricas;
- Jupyter: organização reprodutível da análise e documentação das decisões.

A abordagem clássica de feature engineering foi escolhida por ser compatível com o pequeno número de gravações independentes, permitir interpretação física das variáveis e reduzir o custo computacional quando comparada a uma abordagem end-to-end de Deep Learning.

## 5. Pipeline

```text
CSV bruto
  -> metadados por nome de arquivo
  -> EDA e qualidade dos sinais
  -> janelamento temporal
  -> features no tempo e frequência
  -> análise de redundância/discriminabilidade
  -> particionamento agrupado por arquivo
  -> tuning nos folds internos
  -> avaliação nos folds externos
  -> métricas e matriz de confusão
```

### 5.1 Análise exploratória

A EDA compara as quatro classes sob condições operacionais controladas e também avalia o efeito de velocidade e carga dentro de cada classe. 
Foram analisados:

- sinais no domínio do tempo;
- espectro obtido por FFT;
- RMS, peak-to-peak, kurtosis e crest factor;
- tendência temporal do RMS;
- valores ausentes e um indicador preliminar de clipping/saturação.

A análise mostrou forte influência da velocidade de operação sobre as assinaturas vibroacústicas, além de diferenças entre motor saudável e condições de falha. O Acelerômetro 1 exige interpretação adicional por estar sujeito a maior interferência elétrica do VFD.

### 5.2 Janelamento

Cada gravação de 10 s é segmentada em janelas de 1 s, com 50% de sobreposição. Como a frequência de amostragem é 42 kHz, cada janela contém 42.000 amostras por canal e uma nova janela é iniciada a cada 0,5 s.

Essa configuração produz:

- 19 janelas por arquivo;
- 304 janelas no total;
- 76 janelas por classe.

A janela de 1 s fornece resolução espectral de aproximadamente 1 Hz e mantém duração suficiente para representar componentes de baixa frequência associadas às velocidades nominais de 15 e 45 Hz. A sobreposição de 50% aumenta a resolução temporal sem reduzir a duração utilizada no cálculo espectral.

As janelas extraídas de uma mesma gravação não são observações independentes, pois compartilham a mesma condição de operação e a mesma condição do motor. Por esse motivo, todas as janelas provenientes de um mesmo arquivo são mantidas no mesmo fold durante a validação, evitando que informações da mesma gravação apareçam simultaneamente nos conjuntos de treino e teste.

### 5.3 Feature engineering

São calculadas 14 features por canal, totalizando 56 features candidata para os quatro sinais que estamos trabalhando.

Features no domínio do tempo:

- média;
- desvio padrão;
- RMS;
- amplitude peak-to-peak;
- skewness;
- kurtosis;
- crest factor.

Features no domínio da frequência:

- magnitude próxima de 1x, 2x e 3x da frequência nominal;
- energia espectral;
- centroide espectral;
- entropia espectral;
- frequência dominante.

A FFT é calculada após remoção da média, utilizando as frequências positivas. As features espectrais são limitadas, por padrão, a 2 kHz. Para os componentes 1x, 2x e 3x, o pico é procurado em uma faixa de ±5% ao redor do múltiplo da frequência nominal do VFD, reduzindo a sensibilidade ao deslocamento causado pelo escorregamento do motor de indução.

### 5.4 Discriminabilidade e redundância

A capacidade discriminativa das features é explorada com:

- ANOVA F-value;
- informação mútua.

A redundância é avaliada por correlação de Spearman. No conjunto extraído foram encontrados 78 pares com |correlação| >= 0,90. A seleção atual ordena as features pela informação mútua e remove variáveis fortemente correlacionadas com alguma feature já mantida, reduzindo o conjunto de 56 para 34 features.

Entre as features mais discriminativas aparecem variáveis dos três acelerômetros e do microfone, tanto no domínio temporal quanto espectral. Isso indica que a informação útil não está concentrada em um único sensor ou representação.

## 6. Estratégia de validação

A principal restrição metodológica é impedir que janelas provenientes do mesmo arquivo apareçam simultaneamente em treino e teste. Caso isso ocorresse, o modelo poderia aprender características específicas da gravação em vez de generalizar para uma aquisição não observada.

Foi utilizado `StratifiedGroupKFold`, com `filename` como grupo e `class_name` para estratificação, em uma estratégia nested:

- outer K = 4: avaliação final;
- inner K = 3: escolha dos hiperparâmetros.

Cada fold externo mantém quatro arquivos para teste, um por classe, enquanto os outros 12 arquivos são utilizados no treinamento. Os folds internos operam apenas sobre os arquivos do conjunto de treino externo.

O código executa verificações explícitas para garantir que nenhum `filename` esteja simultaneamente nos lados de treino e validação/teste.

O `StandardScaler` utilizado pelo SVM é ajustado apenas com os dados de treino de cada ajuste e depois aplicado ao conjunto de avaliação correspondente.

## 7. Modelos

Foram comparados dois classificadores com mecanismos de decisão distintos.

### Random Forest

Escolhido por lidar bem com relações não lineares e interações entre features, não exigir padronização e oferecer uma referência robusta para dados tabulares.

Grade de hiperparâmetros:

- `n_estimators`: 100, 300;
- `max_depth`: None, 8, 16;
- `min_samples_leaf`: 1, 2, 4;
- `class_weight="balanced"`;
- `random_state=42`.

### SVM com kernel RBF

Escolhido como modelo complementar baseado em fronteiras não lineares no espaço de features. As features são padronizadas antes do treinamento.

Grade de hiperparâmetros:

- `C`: 0.1, 1, 10, 100;
- `gamma`: `scale`, 0.001, 0.01, 0.1;
- `class_weight="balanced"`;
- `random_state=42`.

As probabilidades do SVM são obtidas com calibração sigmoidal (`CalibratedClassifierCV`, `cv=3`). Essa calibração ocorre somente sobre os dados disponibilizados ao treinamento do modelo externo, portanto não acessa o fold externo de teste. Entretanto, a CV interna do calibrador não é agrupada por arquivo; em uma versão de produção ou em uma avaliação probabilística mais rigorosa, a calibração também deveria respeitar os grupos.

A combinação de hiperparâmetros é escolhida pela maior média de F1-macro nos folds internos.

## 8. Métricas

Foram utilizadas:

- Accuracy;
- F1-macro;
- Precision-macro;
- Recall-macro;
- ROC-AUC macro, one-vs-rest;
- PR-AUC macro, one-vs-rest;
- precision, recall e F1 por classe;
- matriz de confusão agregada dos folds externos.

O F1-macro é utilizado como principal critério de tuning porque atribui o mesmo peso às quatro classes e exige simultaneamente boa precisão e sensibilidade. Mesmo com classes balanceadas, ele é mais informativo do que accuracy quando o objetivo é evitar que bom desempenho em uma condição oculte falhas graves em outra.

A PR-AUC complementa a avaliação por medir o compromisso entre precisão e recall em cada classe no esquema one-vs-rest. Em manutenção preditiva, isso é relevante porque os custos de alarmes falsos e de falhas não detectadas são diferentes. A ROC-AUC mede capacidade de ranqueamento, mas pode permanecer moderada mesmo quando a decisão final por classe ainda é insuficiente; por isso ela deve ser interpretada junto ao F1 e à matriz de confusão.

## 9. Resultados

Resultados médios nos quatro folds externos:

| Modelo | Accuracy | F1-macro | Precision-macro | Recall-macro | ROC-AUC macro | PR-AUC macro |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.401 ± 0.088 | 0.376 ± 0.081 | 0.365 ± 0.087 | 0.401 ± 0.088 | 0.637 ± 0.127 | 0.517 ± 0.102 |
| SVM RBF | 0.365 ± 0.116 | 0.315 ± 0.065 | 0.296 ± 0.046 | 0.365 ± 0.116 | 0.597 ± 0.074 | 0.429 ± 0.024 |

O Random Forest apresentou desempenho médio superior ao SVM em todas as métricas agregadas avaliadas, mas a separação entre os três tipos de falha permaneceu limitada.

F1 por classe nas predições concatenadas dos folds externos:

| Classe | Random Forest | SVM RBF |
|---|---:|---:|
| Healthy | 1.000 | 1.000 |
| Unbalance | 0.280 | 0.213 |
| Misalignment | 0.090 | 0.228 |
| Bowed rotor | 0.242 | 0.000 |

O resultado mais claro é a separação do estado Healthy, classificado corretamente em todas as janelas de teste agregadas. Em contraste, há forte confusão entre Unbalance, Misalignment e Bowed rotor. Assim, o pipeline atual demonstra maior capacidade para distinguir saudável versus falha do que para identificar de forma confiável o subtipo da falha.

Esse comportamento sugere duas linhas de evolução: aumentar a diversidade de gravações independentes para classificação multiclasse ou avaliar uma arquitetura hierárquica, com uma primeira etapa Healthy/Fault e uma segunda etapa dedicada ao diagnóstico do tipo de falha.

### 9.1 Trade-off industrial

Os erros têm custos assimétricos:

- falso positivo de falha: pode provocar inspeção, manutenção ou parada desnecessária de uma máquina saudável;
- falso negativo de falha: pode permitir a operação de um ativo defeituoso, aumentando risco de degradação, dano secundário e indisponibilidade não planejada;
- confusão entre tipos de falha: pode direcionar a manutenção para o componente ou procedimento incorreto.

Por esse motivo, a escolha do limiar e da estratégia de decisão em produção não deveria ser baseada apenas em accuracy. Seria necessário associar as métricas a uma matriz de custos operacionais e, em cenários críticos, privilegiar recall para condições de falha sem aceitar uma taxa excessiva de falsos alarmes.

## 10. Limitações

As principais limitações são:

1. Poucas gravações independentes: existem apenas 16 arquivos, apesar de o janelamento gerar 304 linhas para modelagem. As janelas de um mesmo arquivo são altamente relacionadas.
2. Cobertura operacional restrita: há apenas duas velocidades e duas condições de carga, o que limita a extrapolação para regimes intermediários ou diferentes.
3. Falhas induzidas em bancada: o comportamento pode diferir de falhas progressivas observadas em operação industrial real.
4. Seleção de features global: a implementação atual utiliza todas as janelas para ranquear e selecionar features antes da nested CV, introduzindo informação do teste na decisão de features.
5. Sobreposição entre janelas: janelas adjacentes compartilham 50% das amostras. O agrupamento por arquivo impede vazamento entre treino e teste, mas não cria novas observações independentes.
6. Velocidade mecânica não medida diretamente: as features harmônicas utilizam a frequência nominal do VFD. O escorregamento pode deslocar a frequência real de rotação.
7. Acelerômetro 1 sujeito a ruído do VFD: parte das features discriminativas desse sensor pode refletir a condição elétrica/operacional, e não apenas a falha mecânica.

## 11. Evolução para um sistema real

### 11.1 Batch para streaming

A estratégia atual pode ser convertida para streaming mantendo o mesmo conceito de janelas:

```text
Aquisição contínua
  -> buffer de 1 s
  -> avanço de 0,5 s
  -> validação do sinal
  -> extração das features
  -> inferência
  -> agregação temporal das decisões
  -> alerta / armazenamento / monitoramento
```

Com a configuração atual, uma nova inferência pode ser produzida aproximadamente a cada 0,5 s. Em produção, a decisão de manutenção não deveria ser baseada em uma única janela. Seria recomendável exigir persistência da condição em várias janelas consecutivas ou utilizar uma regra de suavização temporal.


### 11.2 Feature engineering clássico vs. Deep Learning

A abordagem clássica é adequada quando:

- o número de gravações independentes é pequeno;
- interpretabilidade é importante;
- há conhecimento físico útil sobre frequências, energia e impulsividade;
- os recursos de inferência são limitados.

Modelos end-to-end de Deep Learning passam a ser mais justificáveis com um conjunto muito maior e mais diverso de motores, cargas, velocidades, severidades e estágios de degradação. Nesse cenário, CNNs ou arquiteturas temporais poderiam aprender representações diretamente do sinal ou de espectrogramas, mas exigiriam maior custo computacional, validação rigorosa por ativo e mecanismos adicionais de interpretabilidade.

### 11.3 Deploy em edge

A solução baseada em features e modelos clássicos é compatível com edge computing porque o vetor final é pequeno e a inferência dos modelos é relativamente barata. O principal custo está no processamento dos sinais, especialmente na FFT de quatro canais a 42 kHz.

Em hardware restrito, uma evolução possível seria validar filtragem e redução da frequência de amostragem para a banda efetivamente utilizada pelo diagnóstico, preservando Nyquist e evitando aliasing. Isso reduziria CPU, memória e transferência de dados, mas só deveria ser aplicado após demonstrar que as componentes descartadas não carregam informação diagnóstica relevante.

## 12. Reprodutibilidade

### 12.1 Criar o ambiente

```bash
cd <diretorio-do-repositorio>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 12.2 Executar o notebook

1. Coloque os 16 CSVs do subset em um diretório local.
2. Abra `main.ipynb`.
3. Altere `DATA_DIR` para o caminho do dataset:

```python
DATA_DIR = Path("/caminho/para/dataset_desafio")
```

4. Execute as células em ordem, do início ao fim.

O pipeline utiliza `random_state=42` nas etapas estocásticas relevantes. A estrutura dos arquivos deve respeitar a nomenclatura descrita no dicionário de dados.

Para registrar o ambiente como kernel Jupyter, se necessário:

```bash
./env/bin/python -m ipykernel install --user --name=fiepe-env --display-name "Python (FIEPE env)"
```

## 13. Referências

- Sehri, M.; Dumond, P.; Bouchard, M. *University of Ottawa constant and variable speed electric motor vibration and acoustic fault signature dataset*. Data in Brief, 53, 109327, 2024.
- University of Ottawa Electric Motor Dataset, Mendeley Data. DOI: `10.17632/msxs4vj48g`.
- Tiboni, M.; Remino, C.; Bussola, R.; Amici, C. *A Review on Vibration-Based Condition Monitoring of Rotating Machinery*. Applied Sciences, 12(3), 972, 2022. DOI: `10.3390/app12030972`.
