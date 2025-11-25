# Prototipo de restauracion de imagágenes digitales

Este repositorio contiene la implementación del prototipo dashboard de Voila desde GitHub. [Hugging Face Spaces](https://huggingface.co/spaces). 

![vl-hf](https://github.com/voila-dashboards/voila-huggingface/assets/4451292/48464930-c657-4a36-9f00-5eea576f956d)

## Links.

- Hugging Face Project: <https://huggingface.co/spaces/voila-dashboards/voila-huggingface>
- Web App: <https://voila-dashboards-voila-huggingface.hf.space/>
- Spaces Documentation: <https://huggingface.co/docs/hub/spaces>

## Implementacion del dashboard.
1. Crear una cuenta en [Hugging Face](https://huggingface.co/) y genere un token de acceso de usuario con derechos de escritura.

2. Cree un nuevo repositorio en GitHub utilizando este repositorio como plantilla.

3. Crea un nuevo espacio en blanco en Docker Space en Hugging Face.

5. Remplace al URL del proyecto definida en `.github/workflows/update-hf.yml` (current URL is `huggingface.co/spaces/voila-dashboards/voila-huggingface`) con la URL de su proyecto Space.

6. Agregue 2 secretos a este repositorio, llamados `HF_USER` para el nombre de usuario Hugging Face y `HF_TOKEN` para el token de acceso.

7. Añade nuevos notebooks al derectorio `notebooks`.

8. Actualice las dependencias del dashboard en el archivo `environment.yml`.

9. En cada configuración a la rama `main`, el repositorio se sincronizará y se implementará en Hugging Face Spaces.

10. Para obtener acceso directo a su implementación, vaya a `Embed this Space` y copie la URL directa del espacio.

![hf1](https://github.com/voila-dashboards/voila-huggingface/assets/4451292/7af28013-617b-46c5-a07d-16e885a5581f)
![hf-2](https://github.com/voila-dashboards/voila-huggingface/assets/4451292/5d685fe9-45c8-4f77-9f0c-da6686dde09f)
