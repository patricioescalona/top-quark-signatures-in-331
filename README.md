# Top Quark Signatures in 331

Este repositorio guarda los archivos del modelo y el script de barrido en MadGraph.

## Estructura

- `model-generation/`
  - `top-pseudoscalar.fr`: implementación del modelo en FeynRules.
  - `load-top-pseudoscalar.nb`: notebook de Mathematica para cargar el modelo y exportar el UFO.
  - `top-pseudoscalar_UFO/`: copia de referencia para bookkeeping. El flujo real debe funcionar en la instalación local de FeynRules/MadGraph, no desde esta carpeta del repo.
- `scan-in-madgraph/`
  - `cpodd-mass-vs-cs.py`: script para barrer masas y guardar la tabla final de secciones eficaces.
- `results/`
  - tablas finales producidas por el script.

## Flujo completo

### 1. Preparar el modelo en la instalación local de FeynRules

Los archivos de `model-generation/` se conservan aquí como referencia y control de versión.

La exportación del UFO debe hacerse en la instalación local de FeynRules. El flujo esperado es:

1. Copiar o usar allí `top-pseudoscalar.fr` y `load-top-pseudoscalar.nb`.
2. Abrir Mathematica en la instalación local de FeynRules.
3. Cargar el modelo y exportar el UFO.
4. Mover el UFO exportado a la ubicación local donde MadGraph lo pueda importar.

Este repositorio no compila FeynRules ni exporta el UFO por sí mismo.

### 2. Preparar el proceso una vez en la terminal local de MadGraph

En la instalación local de MadGraph, por ejemplo en:

```bash
cd ~/Documents/mg5amcnlo-3.x/bin
./mg5_aMC
```

Luego, dentro de la terminal de MadGraph:

```text
import model top-pseudoscalar_UFO
generate p p > h3 t
output top-pseudoscalar-test
```

Notas:

- `top-pseudoscalar_UFO` es el nombre del modelo UFO ya exportado y ubicado localmente.
- `top-pseudoscalar-test` es el nombre del proceso generado en MadGraph.
- Esa preparación se hace una vez, o cada vez que cambie el proceso.

### 3. Ejecutar el barrido desde este repositorio

Desde la raíz de este repo:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py
```

Por defecto, el script asume:

- instalación local de MadGraph en `~/Documents/mg5amcnlo-3.x/bin`
- proceso local llamado `top-pseudoscalar-test`
- partícula con PDG `36`
- barrido de masas `500, 1000, 1500`
- salida en `results/top-pseudoscalar-scan.csv`

## Qué hace el script

El script:

1. Usa la instalación local de MadGraph.
2. Busca un proceso ya generado dentro de `mg5amcnlo-3.x/bin`.
3. Lanza una corrida por cada masa.
4. Lee la sección eficaz y su error desde los archivos producidos por MadGraph.
5. Guarda solo una tabla resumen en `results/`.

El script no:

- exporta el UFO;
- recompila FeynRules;
- rehace `import model` o `generate`;
- copia al repo los logs grandes de MadGraph.

Los archivos pesados de cada corrida quedan en la carpeta local de MadGraph, no duplicados aquí.

## Opciones principales

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py --help
```

Opciones útiles:

- `--mg5-bin`: ruta a la carpeta `bin` de la instalación local de MadGraph.
- `--process`: nombre del proceso ya generado en MadGraph.
- `--pdg`: PDG de la partícula cuya masa se modifica.
- `--output-name`: nombre base del archivo de salida en `results/`.
- `--output-format`: `csv`, `tsv` o `json`.
- `--masses`: lista explícita de masas.
- `--mass-start`, `--mass-stop`, `--mass-step`: malla regular de masas.
- `--dry-run`: muestra la configuración sin lanzar MadGraph.
- `--stop-on-error`: corta el barrido si falla un punto.

## Ejemplos

Barrido con malla regular:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --output-name top-pseudoscalar-h3t \
  --mass-start 500 \
  --mass-stop 1500 \
  --mass-step 100
```

Barrido con lista explícita:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --output-name top-pseudoscalar-h3t \
  --masses 500 750 1000 1250 1500
```

Usando otro proceso ya preparado en MadGraph:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --process otro-proceso-local \
  --output-name scan-otro-proceso
```

## Diferencia entre `--process` y `--output-name`

- `--process` debe coincidir con el nombre del directorio del proceso ya creado en la terminal local de MadGraph.
- `--output-name` solo cambia el nombre del archivo resumen que se guarda en `results/`.

Ejemplo:

- si en MadGraph hiciste `output top-pseudoscalar-test`, entonces `--process` debe ser `top-pseudoscalar-test`;
- si usas `--output-name top-pseudoscalar-h3t`, el script guardará `results/top-pseudoscalar-h3t.csv`.

## Salida

La tabla final guarda:

- `mass`
- `run_name`
- `cross_section_pb`
- `cross_section_error_pb`
- `status`
- `return_code`
- `note`

## Limpieza

El repo está configurado para no versionar:

- cachés de Python;
- carpetas de logs temporales dentro de `results/`;
- archivos temporales de editor comunes.

Si MadGraph genera artefactos nuevos dentro de su instalación local, esos quedan fuera del repo.
