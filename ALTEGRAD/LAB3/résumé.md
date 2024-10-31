**Qu'est-ce que le pré-entraînement des LLMs ?**

Le pré-entraînement des LLMs consiste à entraîner un modèle sur de vastes corpus de données textuelles non étiquetées afin de lui apprendre à comprendre et générer du texte. Ce processus repose sur des architectures neuronales profondes, comme les réseaux de neurones transformers, qui sont particulièrement efficaces pour capturer les relations entre les mots et les concepts dans les données textuelles.

Le pré-entraînement est souvent suivi d'un affinage (fine-tuning), où le modèle est ajusté sur des tâches spécifiques avec des jeux de données annotés plus petits.

L’objectif du pré-entraînement est de créer un modèle qui comprend la structure et les nuances du langage de manière générale, sans avoir besoin d’exemples spécifiques pour chaque tâche. Le modèle apprend à prédire des mots manquants, à comprendre le contexte, et à capturer des relations entre des phrases entières, ce qui permet de réaliser des tâches variées comme la traduction, la réponse à des questions, et la génération de texte.


**Étapes du pré-entraînement**

- Collection de données : Le pré-entraînement nécessite un large corpus de textes. Cela inclut souvent des données provenant de sources telles que des livres, des sites Web, des articles de recherche, des réseaux sociaux, etc.

- Masquage et objectif d'entraînement : Pendant le pré-entraînement, le modèle essaie généralement de résoudre une tâche de *prédiction de texte masqué*, où un pourcentage de mots ou de tokens sont masqués, et le modèle doit prédire ces tokens à partir du contexte. C'est ce qu'on appelle souvent l'objectif de modélisation de langage masqué (MLM), comme dans BERT. D'autres objectifs peuvent inclure la génération de texte suivant une phrase précédente (comme GPT).

- Utilisation d'architectures avancées : Les LLMs s'appuient souvent sur des architectures comme le transformer, qui permet d'apprendre à partir de séquences longues en modélisant les dépendances à longue distance entre les mots. Cela contraste avec les architectures traditionnelles comme les RNNs et LSTMs.

- Grande échelle : Les LLMs comme GPT-4, BERT, et T5 ont des milliards de paramètres et sont pré-entraînés sur des corpus contenant des centaines de gigaoctets de données textuelles. Cela nécessite d’énormes ressources de calcul et peut durer plusieurs jours ou semaines sur des clusters de GPU.


**Examples : **

• Encoder Only (Bert)
        Exemple : Classification de sentiments, analyse d'opinions, détection de spam, catégorisation de texte. L'encodeur de BERT capture le contexte de manière bidirectionnelle (avant et après le mot en question), ce qui permet une compréhension plus fine du sens des mots et de leurs relations au sein de la phrase. BERT permet une meilleure interprétation du texte en traitant toute la séquence d'entrée d'un coup.
• Encoder Decoder (T5)
        Quand utiliser seulement un décodeur ? Lorsque l'objectif est de générer du texte ou de transformer une séquence d'entrée en une séquence de sortie, souvent en mode séquence-à-séquence (seq2seq). 
• Decoder Only (GPT)

**Scaling laws**

Les scaling laws pour les modèles de langage (Large Language Models, LLM) sont des lois empiriques qui décrivent comment les performances des modèles de langage évoluent en fonction de leur taille, de la quantité de données d'entraînement et des ressources de calcul utilisées. Elles permettent de comprendre comment les performances des modèles s'améliorent lorsqu'on augmente l'échelle (le "scaling") de différents facteurs, tels que :

- La **taille du modèle** (nombre de paramètres) : Augmenter le nombre de paramètres dans un modèle (la taille du réseau de neurones) améliore généralement ses performances. 
- La **quantité de données d'entraînement** : performances d'un LLM s'améliorent également à mesure qu'on augmente la quantité de données d'entraînement
- La **puissance de calcul** utilisée (nombre de GPU/TPU et temps d'entraînement) : Plus le modèle est grand et plus les données sont nombreuses, plus il faut de ressources de calcul

Un article clé qui a introduit les scaling laws est celui d'OpenAI sur les modèles de langage, intitulé "*Scaling Laws for Neural Language Models*" (Kaplan et al., 2020). Voici quelques conclusions majeures issues de cette étude :

- La **perte** suit une relation en loi de puissance avec la taille du modèle, la quantité de données et les ressources de calcul. Cela signifie que la diminution de l'erreur suit une décroissance prévisible lorsque ces paramètres augmentent.

- Il existe un point où les diminishing returns commencent : c’est-à-dire que les gains en performances deviennent de moins en moins significatifs à mesure que les augmentations des ressources (modèle, données, calcul) continuent.


**Loi de Zipf** : La loi énonce que la fréquence d'un mot dans un corpus de texte est inversement proportionnelle à son rang dans la liste des mots classés par fréquence décroissante. 
Si on classe les mots d'un corpus selon leur fréquence d'apparition, la loi de Zipf prédit que le n-ième mot le plus fréquent apparaîtra avec une fréquence proportionnelle à $\frac{1}{n}$ 1 Autrement dit, si le mot le plus fréquent apparaît $f_1$ fois, alors le second mot apparaîtra environ $\frac{f_1}{2}$ fois, le troisième $\frac{f_1}{3}$ fois, et ainsi de suite.


**pad_token** : Le pad_token (token de remplissage ou padding) est un token spécial utilisé dans le traitement du langage naturel (NLP) pour gérer des séquences de différentes longueurs dans des tâches d'apprentissage automatique. 
Supposons que vous ayez deux phrases à traiter :

- Phrase 1 : "Je cours"
- Phrase 2 : "Je cours très vite"

Pour les traiter en un batch, elles doivent avoir la même longueur. Si la deuxième phrase a 4 tokens et la première en a seulement 2, vous pouvez utiliser un pad_token pour compléter la première phrase à 4 tokens, comme ceci :

- Phrase 1 (après padding) : "Je cours pad_token pad_token"
- Phrase 2 : "Je cours très vite"

Nous : tokenizer.pad_token = tokenizer.eos_token
indique que le pad_token est défini pour être égal au eos_token (End Of Sequence), qui est le token utilisé pour marquer la fin d'une séquence. Cela signifie que le token de fin de séquence (généralement utilisé pour dire au modèle que la phrase est terminée) sera également utilisé pour remplir les séquences plus courtes.