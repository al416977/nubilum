# NUBILUM - Base de IA

## Objetivo

Preparar la base de segmentación semántica de nubes de puntos para el proyecto NUBILUM.

En esta primera fase, el objetivo no es detectar todos los elementos arquitectónicos o MEP, sino validar un pipeline semántico básico sobre escenas interiores.

## Problema a resolver

Dada una nube de puntos 3D de un entorno arquitectónico interior, se quiere asignar una etiqueta semántica a cada punto.

En esta fase inicial, las etiquetas serán:

- other
- wall
- floor
- ceiling

## Tipo de tarea

Segmentación semántica por punto.

Cada punto de la nube debe quedar clasificado en una de las clases definidas.

## Clases iniciales

| id | clase   | descripción |
|----|---------|-------------|
| 0  | other   | resto de puntos o puntos no clasificados |
| 1  | wall    | muros y superficies verticales principales |
| 2  | floor   | suelo |
| 3  | ceiling | techo |

## Motivación de esta primera versión

Se ha elegido una taxonomía reducida para:

- simplificar la anotación inicial
- validar el flujo de datos
- facilitar un primer entrenamiento baseline
- comprobar la viabilidad del sistema antes de añadir clases más complejas

## Modelo baseline previsto

KPConv o un modelo equivalente de segmentación semántica de point clouds.

## Flujo previsto

1. Carga de nube de puntos
2. Preprocesado
3. Preparación/anotación de dataset
4. Entrenamiento del modelo
5. Inferencia semántica
6. Visualización en NUBILUM
7. Filtrado por clases

## Próximas ampliaciones previstas

Una vez validado este esquema básico, se estudiará añadir nuevas clases como:

- pipe
- light_fixture
- cable_tray
- equipment
- column
- door_window