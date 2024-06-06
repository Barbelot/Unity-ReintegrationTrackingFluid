using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

public class ReintegrationSimulation : MonoBehaviour
{
    [Header("Application")]
    public int framerateCap = 0;

    [Header("Simulation")]
    public string id = "Main";
    [Space]
    [Range(1, 20)] public int updatesPerFrame = 1;
    public Vector2Int resolution = new Vector2Int(1920, 1200);
    public Vector2 size = new Vector2(1.92f, 1.2f);
    public float dt = 1.5f;
    public float mass = 1f;
    public float fluidRho = .5f;
    public float diffusion = 1.12f;

    [Header("Global Forces")]
    public Vector2 gravity = new Vector2(0, -4);

    [Header("Obstacles")]
    public Texture obstacleTexture;

    [Header("Compute Shader")]
    public ComputeShader shader;

    [Header("Output")]
    public Material outputMaterial;

    [Header("Debug")]
    public bool reinitialize = false;

    private RenderTexture _bufferA;
    private RenderTexture _bufferB;
    private RenderTexture _bufferC;
    private RenderTexture _output;

    public struct EmitterInternal {
        public float emitterType;
        public Vector2 position;
        public float fluidRadius;
        public float fluidColor;
        public float velocityRadius; 
        public Vector2 velocityDirection;
        public float velocityStrength;
    }

    private static int EMITTERSIZE = 9 * sizeof(float);

    private List<FluidEmitter> _fluidEmitters;
    private EmitterInternal[] _emittersArray;
    private ComputeBuffer _emittersBuffer;
    
    private int _initKernel, _updateBufferAKernel, _updateBufferBKernel, _updateBufferCKernel, _updateOutputKernel;

    // THREADGROUP size need to be the same as the compute shader group size.
    private static int NUMTHREADS = 8;

    private bool _initialized = false;

    #region MonoBehaviour

    private void OnEnable()
    {
        Initialize();
    }

    // Update is called once per frame
    void Update()
    {
        Initialize();

        if (!_initialized)
            return;

        UpdateEmitters();

        for(int i=0; i<updatesPerFrame; i++)
        {
            UpdateSimulation();
        }

        //Debug
        if(reinitialize) { Reinitialize(); reinitialize = false; }
    }

    private void OnDisable()
    {
        Cleanup();
    }

    #endregion

    void Initialize()
    {
        if (_initialized)
            return;

        //Get compute shader kernels
        if (!shader)
        {
            Debug.LogError("Compute shader of Reintegration simulation is null.");
            return;
        }

        _initKernel = shader.FindKernel("Init");
        _updateBufferAKernel = shader.FindKernel("UpdateBufferA");
        _updateBufferBKernel = shader.FindKernel("UpdateBufferB");
        _updateBufferCKernel = shader.FindKernel("UpdateBufferC");
        _updateOutputKernel = shader.FindKernel("UpdateOutput");
        
        //Create Render Textures
        CreateRenderTextures();

        //Initialize
        ApplyGlobalSettings();

        shader.SetVector("_Resolution", new Vector4(resolution.x, resolution.y, 1, 0));

        shader.SetTexture(_initKernel, "_BufferA", _bufferA);
        shader.SetTexture(_initKernel, "_BufferB", _bufferB);
        shader.SetTexture(_initKernel, "_BufferC", _bufferC);
        shader.SetTexture(_initKernel, "_Output", _output);

        shader.Dispatch(_initKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);

        //Bind to output
        outputMaterial.SetTexture("_Output", _output);
        outputMaterial.SetTexture("_BufferA", _bufferA);
        outputMaterial.SetTexture("_BufferB", _bufferB);
        outputMaterial.SetTexture("_BufferC", _bufferC);

        //Cap framerate
        Application.targetFrameRate = framerateCap;

        //Emitters
        _fluidEmitters = new List<FluidEmitter>();
        _emittersArray = new EmitterInternal[1];
        _emittersBuffer = new ComputeBuffer(1, EMITTERSIZE, ComputeBufferType.Structured);

        //Obstacles
        if(obstacleTexture == null)
        {
            obstacleTexture = new Texture2D(4, 4);
        }

        shader.SetTexture(_updateBufferBKernel, "_Obstacles", obstacleTexture);
        shader.SetTexture(_updateOutputKernel, "_Obstacles", obstacleTexture);

        _initialized = true;
    }

    void Cleanup()
    {
        if (!_initialized)
            return;

        _bufferA.Release();
        _bufferB.Release();
        _bufferC.Release();
        _output.Release();

        _emittersBuffer.Release();

        _initialized = false;
    }

    public void Reinitialize()
    {
        ApplyGlobalSettings();

        shader.SetVector("_Resolution", new Vector4(resolution.x, resolution.y, 1, 0));

        shader.SetTexture(_initKernel, "_BufferA", _bufferA);
        shader.SetTexture(_initKernel, "_BufferB", _bufferB);
        shader.SetTexture(_initKernel, "_BufferC", _bufferC);
        shader.SetTexture(_initKernel, "_Output", _output);

        shader.Dispatch(_initKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);
    }

    void CreateRenderTextures()
    {
        _bufferA = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        _bufferA.enableRandomWrite = true;
        _bufferA.Create();

        _bufferB = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        _bufferB.enableRandomWrite = true;
        _bufferB.Create();

        _bufferC = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        _bufferC.enableRandomWrite = true;
        _bufferC.Create();

        _output = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        _output.enableRandomWrite = true;
        _output.Create();
    }

    void UpdateSimulation()
    {
        ApplyGlobalSettings();

        shader.SetTexture(_updateBufferAKernel, "_BufferA", _bufferA);
        shader.SetTexture(_updateBufferAKernel, "_BufferB", _bufferB);
        shader.Dispatch(_updateBufferAKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);

        shader.SetTexture(_updateBufferBKernel, "_BufferA", _bufferA);
        shader.SetTexture(_updateBufferBKernel, "_BufferB", _bufferB);
        shader.Dispatch(_updateBufferBKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);

        shader.SetTexture(_updateBufferCKernel, "_BufferA", _bufferA);
        shader.SetTexture(_updateBufferCKernel, "_BufferC", _bufferC);
        shader.Dispatch(_updateBufferCKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);

        shader.SetTexture(_updateOutputKernel, "_BufferA", _bufferA);
        shader.SetTexture(_updateOutputKernel, "_BufferB", _bufferB);
        shader.SetTexture(_updateOutputKernel, "_BufferC", _bufferC);
        shader.SetTexture(_updateOutputKernel, "_Output", _output);
        shader.Dispatch(_updateOutputKernel, resolution.x / NUMTHREADS, resolution.y / NUMTHREADS, 1);
    }

    void ApplyGlobalSettings()
    {
        shader.SetFloat("_Time", Time.time);
        shader.SetFloat("_Dt", dt);
        shader.SetFloat("_Mass", mass);
        shader.SetFloat("_FluidRho", fluidRho);
        shader.SetFloat("_Diffusion", diffusion);
        shader.SetVector("_Gravity", new Vector4(gravity.x * .0001f, gravity.y * .0001f, 0, 0));
    }

    #region Emitters

    public void AddEmitter(FluidEmitter emitter)
    {
        if(!_initialized)
            Initialize();

        _fluidEmitters.Add(emitter);

        UpdateEmitters();
    }

    public void RemoveEmitter(FluidEmitter emitter)
    {
        if (!_initialized)
            Initialize();

        if (_fluidEmitters.Contains(emitter))
        {
            _fluidEmitters.Remove(emitter);

            UpdateEmitters();
        }
    }

    void UpdateEmitters()
    {
        int emittersCount = _fluidEmitters.Count;

        if (emittersCount > _emittersArray.Length || emittersCount > _emittersBuffer.count)
        {
            _emittersArray = new EmitterInternal[emittersCount];
            _emittersBuffer.Release();
            _emittersBuffer = new ComputeBuffer(emittersCount, EMITTERSIZE, ComputeBufferType.Structured);
        }

        for (int i=0; i< emittersCount; i++)
        {
            _emittersArray[i].position = GetEmitterUV(_fluidEmitters[i].transform.position);
            _emittersArray[i].velocityStrength = _fluidEmitters[i].emitVelocity ? _fluidEmitters[i].velocityStrength * Time.deltaTime : 0;
            _emittersArray[i].velocityDirection = _fluidEmitters[i].velocityDirection;
            _emittersArray[i].fluidRadius = _fluidEmitters[i].emitFluid ? _fluidEmitters[i].fluidRadius : 0;
            _emittersArray[i].velocityRadius = _fluidEmitters[i].velocityRadius;
            _emittersArray[i].fluidColor = _fluidEmitters[i].fluidColor;
            _emittersArray[i].emitterType = (float)_fluidEmitters[i].emitterType;
        }

        _emittersBuffer.SetData(_emittersArray);

        shader.SetBuffer(_updateBufferBKernel, "_EmittersBuffer", _emittersBuffer);
        shader.SetInt("_EmittersCount", emittersCount);
    }

    Vector2 GetEmitterUV(Vector3 position)
    {
        return new Vector2(position.x / size.x + 0.5f, position.y / size.y + 0.5f);
    }

    #endregion 
}
