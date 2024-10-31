# 1. Principe général du Transfer Learning

L'idée centrale du transfer learning est que les modèles d'apprentissage automatique, notamment ceux de l'apprentissage profond (deep learning), peuvent réutiliser les caractéristiques ou représentations apprises lors d'une tâche, afin de mieux performer sur une tâche différente mais liée. Par exemple, si un modèle a été entraîné pour reconnaître des objets dans des images à l'aide de millions d'exemples, il pourrait réutiliser les caractéristiques apprises, comme les formes, les textures, ou les contours, pour résoudre une tâche de reconnaissance d'objets dans un ensemble de données différent, mais avec moins d'exemples.

# 2. Types de Transfer Learning

- **Transfert inductif** : Le transfert se fait vers une tâche cible qui diffère de la tâche source. Par exemple, un modèle entraîné sur un ensemble de données général (comme ImageNet) pour la classification d'images peut être ajusté pour reconnaître des images spécifiques dans une application industrielle.

- **Transfert transductif** : Les tâches source et cible sont similaires, mais les domaines de données diffèrent. Par exemple, un modèle de reconnaissance vocale entraîné sur des enregistrements en anglais pourrait être adapté pour une autre langue, même si les tâches sont fondamentalement les mêmes (reconnaissance vocale).

- **Transfert non supervisé** : Ici, ni la tâche source ni la tâche cible n'ont d’étiquettes dans l'ensemble des données, mais le modèle peut quand même transférer des représentations entre ces ensembles de données non étiquetés pour accomplir des tâches comme le clustering ou la réduction de dimensions.

# 3. Applications 

- **Vision par ordinateur** : Le transfer learning est particulièrement utilisé dans les tâches de classification d'images, de détection d'objets ou de segmentation d'images. Par exemple, les architectures de réseaux de neurones convolutionnels (CNN) comme ResNet, VGG ou Inception sont souvent pré-entraînées sur de grands ensembles de données (comme ImageNet) puis ajustées pour des tâches spécifiques avec de petits ensembles de données.

- **NLP** : Dans le domaine NLP, des modèles comme BERT, GPT-3 et T5 sont d'abord pré-entraînés sur de larges corpus de texte, puis adaptés à des tâches plus spécifiques comme la classification de texte, la traduction, ou la génération de texte avec une phase de fine-tuning.

- **Reconnaissance vocale et audio** : Les modèles pré-entraînés pour des tâches comme la reconnaissance vocale peuvent être adaptés pour reconnaître des accents spécifiques, de nouveaux langages, ou pour la transcription de conversations dans un domaine spécifique.

- **Robotique et contrôle** : En robotique, le transfer learning est utilisé pour que des robots puissent réutiliser des comportements appris dans des environnements simulés pour opérer dans des environnements réels, réduisant ainsi les coûts de formation et d'adaptation.

# 4. Méthodologies 

- **Fine-tuning** : Le modèle pré-entraîné est réentraîné (partiellement ou complètement) sur une nouvelle tâche avec des nouvelles données. Les couches profondes du réseau sont souvent gelées (leurs poids sont fixés), tandis que les dernières couches sont ajustées pour la tâche cible.

- **Feature extraction** : Le modèle pré-entraîné est utilisé tel quel pour extraire des caractéristiques pertinentes qui sont ensuite utilisées par un autre algorithme (par exemple un classificateur simple comme SVM ou un réseau neuronal à faible profondeur) pour effectuer la nouvelle tâche.

- **Domain adaptation** : Cette méthode tente d'aligner les distributions des données source et cible, en ajustant les représentations apprises pour s'adapter au nouveau domaine, surtout lorsque les données de la tâche cible sont peu nombreuses ou trop différentes de la source.

# Définitions : 

- La "classification head" (tête de classification) est la dernière couche d'un modèle de réseau de neurones, spécifiquement conçue pour effectuer une tâche de classification. Elle prend en entrée les représentations apprises par le reste du modèle (par exemple, les features extraites par un Transformer ou un réseau convolutif) et produit une prédiction finale sous forme de probabilité pour chaque classe cible.

- **DataLoader** est une classe de PyTorch qui permet de charger un jeu de données et de le diviser en mini-batches, utilisés pendant l'entraînement d'un modèle. Cela aide à entraîner le modèle plus efficacement en traitant plusieurs échantillons de données à la fois.
Les **mini-batches** sont des petits sous-ensembles du jeu de données complet. En traitant un mini-batch à la fois, le modèle peut s'entraîner de manière plus rapide et stable, par rapport à traiter l'ensemble des données en une seule fois (batch complet) ou un seul exemple à la fois (stochastique).