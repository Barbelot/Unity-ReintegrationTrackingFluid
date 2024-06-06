using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FluidEmitter : MonoBehaviour
{
    public enum EmitterType { Directional, Spherical }

    [Header("Simulation")]
    public string simulationId = "Main";

    [Header("Fluid Emitter")]
    public bool emitFluid = true;
    [Tooltip("0 = infinite")] public float emissionDuration = 0;
    [Tooltip("In pixels")] public float fluidRadius = 20;
    public float fluidColor = 0;
    [Space]
    public bool emitVelocity = true;
    public EmitterType emitterType = EmitterType.Directional;
    [Tooltip("In pixels")] public float velocityRadius = 20;
    public float velocityStrength = 1;
    public Vector2 velocityDirection = Vector2.up;

    private ReintegrationSimulation simulation;
    private bool _binded = false;
    private float _emissionStartTime;

    private void OnEnable()
    {
        BindToSimulation();
    }
    private void Update()
    {
        if (!_binded)
            BindToSimulation();

        if (!_binded)
            return;

        //Stop emission after emission duration
        if (emissionDuration > 0 && Time.time - _emissionStartTime > emissionDuration)
        {
            emitFluid = false;
            emitVelocity = false;
        }
    }

    private void OnDisable()
    {
        if (_binded)
        {
            simulation.RemoveEmitter(this);
            _binded = false;
        }
    }

    void BindToSimulation()
    {
        foreach(var s in FindObjectsByType<ReintegrationSimulation>(FindObjectsSortMode.None))
        {
            if(s.id == simulationId)
            {
                simulation = s;
                simulation.AddEmitter(this);
                _binded = true;
                _emissionStartTime = Time.time;
                return;
            }
        }
    }
}
