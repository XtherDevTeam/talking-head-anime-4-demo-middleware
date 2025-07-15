import tha4_api.animation
import tha4_api.api
import livekit.api
import livekit.rtc
import typing
import pathlib
import PIL.Image
import torch
import io
import asyncio
import json
import numpy
import queue
import time
import uuid
import threading
from tha4_api import logger

def lookupCache(cacheName: str) -> typing.Optional[bytes]:
    cachePath = pathlib.Path("tha4_cache") / f'{cacheName}.tha4'
    if cachePath.exists():
        return cachePath
    return None

def compileForConfiguration(configuration: dict, avatar: bytes, baseFps: int = 20) -> None:
    characterConfiguration = tha4_api.animation.AnimationConfiguration(configuration)
    manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))
    manager.set_base_image(PIL.Image.open(io.BytesIO(avatar)))
    renderer = tha4_api.animation.Renderer(characterConfiguration, manager, baseFps=baseFps)
    renderer.compile_all_animations()
    renderer.serailize(f'tha4_cache/{uuid.uuid4().hex}.tha4')

class Live2DRealtimeSession:
    def __init__(self, sessionName: str, configuration: dict, avatar: bytes, baseFps: int = 20):
        self.sessionName = sessionName
        self.connected = False
        self.characterConfiguration = tha4_api.animation.AnimationConfiguration(configuration)
        self.manager = tha4_api.api.ImageInferenceManager(tha4_api.api.ImageInferenceManager.load_model(torch.device('cuda:0')), torch.device('cuda:0'))
        self.manager.set_base_image(PIL.Image.open(io.BytesIO(avatar)))
        self.renderer = tha4_api.animation.Renderer(self.characterConfiguration, self.manager, baseFps)
        self.rendered_frames: queue.Queue[numpy.ndarray] = queue.Queue()
        cacheResult = lookupCache(self.characterConfiguration.name)
        if cacheResult is not None:
            self.renderer.deserialize(str(cacheResult))
            logger.Logger.log(f"Cache found for {self.characterConfiguration.name}, deserializing...")
        else:
            logger.Logger.log(f"No cache found for {self.characterConfiguration.name}, this may cause a significant performance issue while rendering in real-time!...")
        
        self.renderer.on('frame_update', lambda frame: self.rendered_frames.put(frame))
    
            
    async def start(self, live2DToken: str, livekitUrl: str, loop: asyncio.AbstractEventLoop):
        logger.Logger.log(f'Preparing to start live2D session {self.characterConfiguration.name}...')
        self.loop = loop
        self.chatRoom = livekit.rtc.Room(loop)
        self.connected = True
        
        @self.chatRoom.on('participant_connected')
        def on_participant_connected(participant):
            logger.Logger.log(f'Participant {participant.identity} connected to live2D session {self.characterConfiguration.name}...')
        
        @self.chatRoom.on('participant_disconnected')
        def on_participant_disconnected(participant):
            logger.Logger.log(f'Participant {participant.identity} disconnected from live2D session {self.characterConfiguration.name}...')
            
        @self.chatRoom.on('connected')
        def on_connected():
            logger.Logger.log(f'Connected to live2D session {self.characterConfiguration.name}...')
            
        
        await self.chatRoom.connect(livekitUrl, live2DToken)
        logger.Logger.log(f'Connected to livekit server {livekitUrl} with token {live2DToken}...')
        
        videoSource = livekit.rtc.VideoSource(
            512, 512)
        self.broadcastVideoTrack = livekit.rtc.LocalVideoTrack.create_video_track("live2d", videoSource)
        publication_video = await self.chatRoom.local_participant.publish_track(self.broadcastVideoTrack, livekit.rtc.TrackPublishOptions(
            source=livekit.rtc.TrackSource.SOURCE_CAMERA,
            red=False,
            video_encoding=livekit.rtc.VideoEncoding(max_framerate=self.renderer.baseFps, max_bitrate=3000000)
        ))
        logger.Logger.log(f'Published video track to livekit server...')
        asyncio.ensure_future(self.broadcastVideoLoop(videoSource))
        logger.Logger.log(f'Starting renderer loop for live2D session {self.characterConfiguration.name}...')
        self.renderer.start_render_loop()
        
        
    async def broadcastVideoLoop(self, source: livekit.rtc.VideoSource):
        logger.Logger.log(f'Starting video broadcast loop for live2D session {self.characterConfiguration.name}...')
        last_sec = 0
        last_time = time.time()
        while self.connected:
            if time.time() - last_time > 1:
                logger.Logger.log(f'Rendered {last_sec} frames for live2D session {self.characterConfiguration.name}...')
                last_sec = 0
                last_time = time.time()
            else:
                last_sec += 1
            frame = self.rendered_frames.get()
            livekitFrame = livekit.rtc.VideoFrame(512, 512, livekit.rtc.VideoBufferType.RGBA, frame)
            source.capture_frame(livekitFrame)
            time.sleep(1/self.renderer.baseFps)
        
        logger.Logger.log(f'Stopping video broadcast loop for live2D session {self.characterConfiguration.name}...')

    async def shutdown(self):
        self.connected = False
        await self.chatRoom.disconnect()
        self.renderer.stop_render_loop()
        
        
class Live2DRealtimeSessionManager:
    def __init__(self):
        self.sessions: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        
    def createSession(self, configuration: dict, avatar: bytes, baseFps: int = 20) -> str:
        sessionName = uuid.uuid4().hex
        session = Live2DRealtimeSession(sessionName, configuration, avatar, baseFps)
        self.sessions[sessionName] = {
            'session': session,
            'last_access_time': time.time()
        }
        return sessionName

    def getSession(self, sessionName: str, refresh_access_time: bool = True) -> typing.Optional[Live2DRealtimeSession]:
        session = self.sessions.get(sessionName)
        if session is not None:
            if refresh_access_time:
                session['last_access_time'] = time.time()
            return session['session']
        return None
    
    def shutdownSession(self, sessionName: str):
        session = self.sessions.get(sessionName)
        if session is not None:
            asyncio.ensure_future(session['session'].shutdown(), session['session'].loop)
            del self.sessions[sessionName]
            
    def daemon_thread_wrapper(self):
        while True:
            time.sleep(60)
            for session in self.sessions.values():
                if time.time() - session['last_access_time'] > 300:
                    self.shutdownSession(session['session'].sessionName)
                    logger.Logger.log(f'Session {session["session"].sessionName} has been idle for 5 minutes, shutting down...')
                    
    def start_daemon_thread(self):
        threading.Thread(target=self.daemon_thread_wrapper, daemon=True).start()