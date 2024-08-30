import abc
import argparse
import base64
import contextlib
import datetime
import functools
import io
import itertools
import logging
import mimetypes
import os
import pathlib
import signal
import subprocess
import threading
import time
import uuid
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import anthropic
import anthropic.types
import av
import elevenlabs
import elevenlabs.client
import gpiozero
import openai
import pasimple
import PIL.Image
import prctl
import tenacity
import xdg_base_dirs
from pydantic import BaseModel, ConfigDict, Field
from to_file_like_obj import to_file_like_obj

from .data import Skill, SkillName, prompt_with_quotes

logger = logging.getLogger(__name__)

__version__: str = __import__("importlib.metadata").metadata.version(__package__ or __name__)


T = TypeVar("T")


class BufferedIterator(Iterable[T]):
    def __init__(self, iterator: Iterator[T]) -> None:
        self._iterator = iterator
        self._buffer: List[T] = []
        self._exhausted = False

    def __iter__(self) -> Generator[T, None, None]:
        for item in self._buffer:
            yield item
        if not self._exhausted:
            for item in self._iterator:
                self._buffer.append(item)
                yield item
            self._exhausted = True


F = TypeVar("F", bound=Callable[..., Any])


def measure_response_time(fn: F) -> F:
    @functools.wraps(fn)
    def _measure_response_time(*args, **kwargs):
        start_time = time.monotonic()
        logger.info("Entering %s", fn.__qualname__)
        try:
            return fn(*args, **kwargs)
        finally:
            logger.info("Exiting %s after %f seconds", fn.__name__, time.monotonic() - start_time)

    return cast(F, _measure_response_time)


class ImageData(NamedTuple):
    data: bytes
    mime_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{self.to_base64()}"


class AudioDataStream(NamedTuple):
    data: BufferedIterator[bytes]
    mime_type: Literal["audio/mpeg"]


class SkillResponse(BaseModel):
    skill: SkillName = Field(description="Which skill voice is responding")
    success: bool = Field(description="Whether the skill check was successful")
    reason: str = Field(
        description="Specific justification for why this skill voice was chosen in this context"
    )
    dialogue: str = Field(
        description="Dialogue produced by the skill voice, not including dialogue tags"
    )
    model_config = ConfigDict(extra="forbid")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=pathlib.Path)
    parser.add_argument("--output-audio", type=pathlib.Path)
    parser.add_argument("--storage-path", type=pathlib.Path, default="./data")
    parser.add_argument("--gpio-pin", type=int, default=None)
    parser.add_argument("--vision-model", choices=["gpt", "claude"], default="gpt")
    return parser.parse_args()


class BaseImageSource(abc.ABC):
    def _image_to_webp(self, image: PIL.Image.Image) -> ImageData:
        with io.BytesIO() as buffer:
            resized = PIL.ImageOps.contain(image.convert("RGB"), (512, 512))
            resized.save(buffer, format="webp", quality=80)
            return ImageData(data=buffer.getvalue(), mime_type="image/webp")

    @abc.abstractmethod
    def capture_image(self) -> ImageData: ...


class FileImageSource(BaseImageSource):
    def __init__(self, image_path: pathlib.Path):
        self._image_path = image_path

    @measure_response_time
    def capture_image(self) -> ImageData:
        with PIL.Image.open(self._image_path) as image:
            return self._image_to_webp(image)


class PiCameraImageSource(BaseImageSource):
    def __init__(self):
        if not (xdg_runtime_dir := xdg_base_dirs.xdg_runtime_dir()):
            raise Exception("No XDG_RUNTIME_DIR")
        storage_path = xdg_runtime_dir / "disco"
        storage_path.mkdir(parents=True, exist_ok=True)
        self._symlink_path = storage_path / "capture.jpg"
        self._process = subprocess.Popen(
            [
                "rpicam-still",
                "--mode=512:512:12:P",
                "--width=1024",
                "--height=512",
                "--timeout=0",
                "--signal",
                "--datetime",
                "--autofocus-speed=fast",
                f"--latest={self._symlink_path.name}",
                "--quality=70",
                "--verbose=0",
            ],
            cwd=storage_path,
            preexec_fn=self._rpicam_preexec_fn,
        )
        self._timeout = 5

    @staticmethod
    def _rpicam_preexec_fn() -> None:
        prctl.set_pdeathsig(signal.SIGKILL)
        # ignore SIGUSR1 until rpicam-still installs its own handler
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)

    @measure_response_time
    def capture_image(self) -> ImageData:
        self._symlink_path.unlink(missing_ok=True)
        os.kill(self._process.pid, signal.SIGUSR1)
        start_time = time.monotonic()
        while not self._symlink_path.exists():
            time.sleep(0.05)
            if time.monotonic() - start_time > self._timeout:
                raise Exception("Timed out while capturing image from camera")
        with PIL.Image.open(self._symlink_path) as image:
            self._symlink_path.resolve().unlink(missing_ok=True)
            return self._image_to_webp(image)


class BaseVisionModel(abc.ABC):
    @abc.abstractmethod
    def respond_to_image(self, image: ImageData) -> Optional[SkillResponse]: ...


class ClaudeVisionModel(BaseVisionModel):
    def __init__(
        self, *, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 1024
    ) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    @measure_response_time
    def respond_to_image(self, image: ImageData) -> Optional[SkillResponse]:
        response = self._client.beta.prompt_caching.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": prompt_with_quotes,
                },
            ],
            tools=[
                {
                    "name": "skill_dialogue",
                    "description": "Respond with a skill voice",
                    "input_schema": SkillResponse.model_json_schema(),
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.mime_type,
                                "data": image.to_base64(),
                            },
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
        )
        for block in response.content:
            if isinstance(block, anthropic.types.ToolUseBlock):
                return SkillResponse.parse_obj(block.input)
        return None


class GPTVisionModel(BaseVisionModel):
    def __init__(self, *, model: str = "gpt-4o-2024-08-06") -> None:
        self._client = openai.OpenAI(timeout=10, max_retries=2)
        self._model = model

    @measure_response_time
    def respond_to_image(self, image: ImageData) -> Optional[SkillResponse]:
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_with_quotes,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image.to_data_uri()},
                        }
                    ],
                },
            ],
            response_format=SkillResponse,
        )
        # TODO what happens on a decline here?
        return response.choices[0].message.parsed


class BaseSpeechModel(abc.ABC):
    @abc.abstractmethod
    def synthesize_skill_response(
        self,
        skill_response: SkillResponse,
    ) -> AudioDataStream: ...


class ElevenlabsSpeechModel(BaseSpeechModel):
    def __init__(self, *, model: str = "eleven_turbo_v2_5") -> None:
        self._client = elevenlabs.client.ElevenLabs(timeout=10)
        self._model = model

    @measure_response_time
    def synthesize_skill_response(
        self,
        skill_response: SkillResponse,
    ) -> AudioDataStream:
        response = self._client.generate(
            model=self._model,
            text=skill_response.dialogue,
            voice=elevenlabs.Voice(
                voice_id=(
                    "ccAh0EIwBmmUxEUQeoWP"
                    if skill_response.skill is SkillName.HORRIFIC_NECKTIE
                    else "JlaQBUx1SH2NEjFCK7Bh"
                ),
                settings=elevenlabs.VoiceSettings(
                    similarity_boost=1.0,
                    stability=0.01,
                    style=0,
                    use_speaker_boost=False,
                ),
            ),
            output_format="mp3_22050_32",
            stream=True,
        )
        return AudioDataStream(BufferedIterator(response), "audio/mpeg")


class BaseTrigger(abc.ABC):
    @abc.abstractmethod
    def wait_for_trigger(self) -> bool: ...


class KeyboardTrigger(BaseTrigger):
    def wait_for_trigger(self) -> bool:
        input("Press enter to trigger")
        return True


class GPIOTrigger(BaseTrigger):
    def __init__(self, pin: int) -> None:
        self._pin = pin
        self._button = gpiozero.Button(pin)

    def wait_for_trigger(self) -> bool:
        logger.info("Waiting for GPIO %d", self._pin)
        self._button.wait_for_press()
        return True


class BaseAudioSink(abc.ABC):
    def _get_asset(self, name: str) -> pathlib.Path:
        return pathlib.Path(__file__).parent / "assets" / name

    def _create_graph(
        self, skill_response: SkillResponse, audio_stream: av.audio.stream.AudioStream
    ) -> av.filter.Graph:
        skill = Skill.from_name(skill_response.skill)
        graph = av.filter.Graph()
        # outcome = graph.add("amovie", str(self._get_asset("success.wav" if skill_response.success else "failure.wav")))
        # anullsink = graph.add("anullsink")
        # outcome.link_to(anullsink)
        if skill.category:
            skillsound = graph.add(
                "amovie", str(self._get_asset(skill.category.value.lower() + ".wav"))
            )
        else:
            skillsound = graph.add("anullsrc")
        skill_forever = graph.add("apad")
        skillsound.link_to(skill_forever)
        abuffer = graph.add_abuffer(template=audio_stream)
        delay = graph.add("adelay", "2000")
        abuffer.link_to(delay)
        buffersink = graph.add("abuffersink")
        aformat = graph.add("aformat", "sample_fmts=s16:channel_layouts=mono:sample_rates=48000")
        aformat.link_to(buffersink)
        amerge = graph.add("amerge", "inputs=2")
        skill_forever.link_to(amerge, 0, 0)
        delay.link_to(amerge, 0, 1)
        amerge.link_to(aformat)
        graph.configure()
        return graph

    def _feed_graph_thread(
        self,
        container: av.container.InputContainer,
        audio_stream: av.audio.AudioStream,
        graph_push: Callable[[av.AudioFrame], None],
    ):
        for packet in container.demux(audio_stream):
            for frame in cast(Iterable[av.AudioFrame], packet.decode()):
                graph_push(frame)

    def _generate_graph_frames(
        self, graph_pull: Callable[[], av.AudioFrame], graph_alive: Callable[[], bool]
    ) -> Generator[av.AudioFrame, None, None]:
        while True:
            try:
                yield graph_pull()
            except av.error.BlockingIOError:
                if not graph_alive():
                    break

    @contextlib.contextmanager
    def _create_and_run_graph(
        self, skill_response: SkillResponse, dialogue_audio: AudioDataStream
    ) -> Generator[Generator[av.AudioFrame, None, None], None, None]:
        with av.open(to_file_like_obj(dialogue_audio.data), "r") as container:
            audio_stream = container.streams.audio[0]
            graph = self._create_graph(skill_response, audio_stream)
            thread = threading.Thread(
                target=self._feed_graph_thread,
                args=(container, audio_stream, graph.push),
                daemon=True,
            )
            thread.start()
            try:
                yield self._generate_graph_frames(
                    cast(Callable[[], av.AudioFrame], graph.pull), thread.is_alive
                )
            finally:
                # thread should already be dead, if not, hard crash is fine
                thread.join(timeout=1)

    @abc.abstractmethod
    def play_skill_response(
        self,
        skill_response: SkillResponse,
        dialogue_audio: AudioDataStream,
    ) -> None: ...


class FileAudioSink(BaseAudioSink):
    def __init__(self, output_path: pathlib.Path) -> None:
        self._output_path = output_path

    @measure_response_time
    def play_skill_response(
        self,
        skill_response: SkillResponse,
        dialogue_audio: AudioDataStream,
    ) -> None:
        raise NotImplementedError()


class SpeakerAudioSink(BaseAudioSink):
    @tenacity.retry(stop=tenacity.stop_after_attempt(3), reraise=True)
    @measure_response_time
    def play_skill_response(
        self,
        skill_response: SkillResponse,
        dialogue_audio: AudioDataStream,
    ) -> None:
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK,
            pasimple.PA_SAMPLE_S16LE,
            1,
            48000,
        ) as pa:
            with self._create_and_run_graph(skill_response, dialogue_audio) as graph_frames:
                for frame in graph_frames:
                    pa.write(frame.to_ndarray().tobytes())
            pa.drain()


class DataStorage:
    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    def create_run_storage(self) -> "RunDataStorage":
        run_id = "_".join(
            (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), str(uuid.uuid4()).replace("-", ""))
        )
        return RunDataStorage(self._base_path / run_id, run_id)


class RunDataStorage:
    def __init__(self, base_path: pathlib.Path, run_id: str) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def save_file(self, filename: str, content: Union[str, bytes]) -> pathlib.Path:
        with (self._base_path / filename).open("wb" if isinstance(content, bytes) else "w") as f:
            f.write(content)
        return self._base_path / filename

    @measure_response_time
    def save_files(
        self, *filename_content_pairs: Tuple[str, Union[str, bytes]]
    ) -> Iterable[pathlib.Path]:
        results = []
        for filename, content in filename_content_pairs:
            results.append(self.save_file(filename, content))
        return results


def run_loop(
    trigger: BaseTrigger,
    data_storage: DataStorage,
    image_source: BaseImageSource,
    vision_model: BaseVisionModel,
    speech_model: BaseSpeechModel,
    audio_sink: BaseAudioSink,
) -> None:
    while trigger.wait_for_trigger():
        run_storage = data_storage.create_run_storage()
        logger.info("Starting run %s", run_storage.run_id)
        image = image_source.capture_image()
        skill_response = vision_model.respond_to_image(image)
        if not skill_response:
            logger.warning("No skill response...")
            continue
        logger.info("Skill response: %r", skill_response)
        dialogue_audio = speech_model.synthesize_skill_response(skill_response)
        audio_sink.play_skill_response(skill_response, dialogue_audio)
        run_storage.save_files(
            (f"image{mimetypes.guess_extension(image.mime_type)}", image.data),
            (
                f"voice{mimetypes.guess_extension(dialogue_audio.mime_type)}",
                b"".join(dialogue_audio.data),
            ),
            ("skill.json", skill_response.model_dump_json(indent=4)),
        )
        logger.info("Run %s complete", run_storage.run_id)


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    data_storage = DataStorage(args.storage_path)
    if args.input_image:
        image_source: BaseImageSource = FileImageSource(args.input_image)
    else:
        image_source = PiCameraImageSource()
    if args.gpio_pin is None:
        trigger: BaseTrigger = KeyboardTrigger()
    else:
        trigger = GPIOTrigger(args.gpio_pin)
    if args.vision_model == "gpt":
        vision_model: BaseVisionModel = GPTVisionModel()
    else:
        vision_model = ClaudeVisionModel()
    speech_model = ElevenlabsSpeechModel()
    if args.output_audio:
        audio_sink: BaseAudioSink = FileAudioSink(args.output_audio)
    else:
        audio_sink = SpeakerAudioSink()
    logger.info("Initialization done")
    run_loop(trigger, data_storage, image_source, vision_model, speech_model, audio_sink)
