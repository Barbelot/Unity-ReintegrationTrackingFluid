// Taken from https://www.shadertoy.com/view/WtfyDj

#define Bf(p) p
#define Bi(p) int2(p)
#define texel(a, p) a.Load(int3(p, 0))

#define PI 3.14159265

#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

#define border_h 5.

float2 R;
float4 Mouse;
float time;
float dt;
float mass;
float fluid_rho;
float dif;
float2 gravity;
float vortex;

float Pf(float2 rho)
{
    //return 0.2*rho.x; //gas
    float GF = 1.; //smoothstep(0.49, 0.5, 1. - rho.y);
    return lerp(0.5 * rho.x, 0.04 * rho.x * (rho.x / fluid_rho - 1.), GF); //water pressure
}

float2x2 Rot(float ang)
{
    return float2x2(cos(ang), -sin(ang), sin(ang), cos(ang));
}

float2 Dir(float ang)
{
    return float2(cos(ang), sin(ang));
}


float sdBox(in float2 p, in float2 b)
{
    float2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float border(float2 p)
{
    float bound = -sdBox(p - R * 0.5, R * float2(0.5, 0.5));
    float box = sdBox(mul(Rot(0. * time), (p - R * float2(0.5, 0.6))), R * float2(0.05, 0.01));
    float drain = -sdBox(p - R * float2(0.5, 0.7), R * float2(1.5, 0.5));
    return max(drain, min(bound, box));
}

#define h 1.
float3 bN(float2 p)
{
    float3 dx = float3(-h, 0, h);
    float4 idx = float4(-1. / h, 0., 1. / h, 0.25);
    float3 r = idx.zyw * border(p + dx.zy)
           + idx.xyw * border(p + dx.xy)
           + idx.yzw * border(p + dx.yz)
           + idx.yxw * border(p + dx.yx);
    return float3(normalize(r.xy), r.z + 1e-4);
}

uint pack(float2 x)
{
    x = 65534.0 * clamp(0.5 * x + 0.5, 0., 1.);
    return uint(round(x.x)) + 65535u * uint(round(x.y));
}

float2 unpack(uint a)
{
    float2 x = float2(a % 65535u, a / 65535u);
    return clamp(x / 65534.0, 0., 1.) * 2.0 - 1.0;
}

float2 decode(float x)
{
    uint X = asuint(x);
    return unpack(X);
}

float encode(float2 x)
{
    uint X = pack(x);
    return asfloat(X);
}

struct particle
{
    float2 X;
    float2 V;
    float2 M;
};
    
particle getParticle(float4 data, float2 pos)
{
    particle P;
    P.X = decode(data.x) + pos;
    P.V = decode(data.y);
    P.M = data.zw;
    return P;
}

float4 saveParticle(particle P, float2 pos)
{
    P.X = clamp(P.X - pos, float2(-0.5, -.5), float2(0.5, .5));
    return float4(encode(P.X), encode(P.V), P.M);
}

float3 hash32(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return frac((p3.xxy + p3.yzz) * p3.zyx);
}

float G(float2 x)
{
    return exp(-dot(x, x));
}

float G0(float2 x)
{
    return exp(-length(x));
}

float3 distribution(float2 x, float2 p, float K)
{
    float2 omin = clamp(x - K * 0.5, p - 0.5, p + 0.5);
    float2 omax = clamp(x + K * 0.5, p - 0.5, p + 0.5);
    return float3(0.5 * (omin + omax), (omax.x - omin.x) * (omax.y - omin.y) / (K * K));
}

/*
float3 distribution(float2 x, float2 p, float K)
{
    float4 aabb0 = float4(p - 0.5, p + 0.5);
    float4 aabb1 = float4(x - K*0.5, x + K*0.5);
    float4 aabbX = float4(max(aabb0.xy, aabb1.xy), min(aabb0.zw, aabb1.zw));
    float2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
    float2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive
    float m = size.x*size.y/(K*K); //relative amount
    //if any of the dimensions are 0 then the mass is 0
    return float3(center, m);
}*/

//diffusion and advection basically
void Reintegration(RWTexture2D<float4> ch, inout particle P, float2 pos)
{
    //basically integral over all updated neighbor distributions
    //that fall inside of this pixel
    //this makes the tracking conservative
    range(i, -2, 2)range(j, -2, 2)
        {
            float2 tpos = pos + float2(i, j);
            float4 data = texel(ch, tpos);
       
            particle P0 = getParticle(data, tpos);
       
            P0.X += P0.V * dt; //integrate position

            float difR = 0.9 + 0.21 * smoothstep(fluid_rho * 0., fluid_rho * 0.333, P0.M.x);
            float3 D = distribution(P0.X, pos, difR);
        //the deposited mass into this cell
            float m = P0.M.x * D.z;
        
        //add weighted by mass
            P.X += D.xy * m;
            P.V += P0.V * m;
            P.M.y += P0.M.y * m;
        
        //add mass
            P.M.x += m;
        }
    
    //normalization
    if (P.M.x != 0.)
    {
        P.X /= P.M.x;
        P.V /= P.M.x;
        P.M.y /= P.M.x;
    }
}

float3 hsv2rgb(in float3 c)
{
    float3 rgb = clamp(abs(fmod(c.x * 6.0 + float3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);

    rgb = rgb * rgb * (3.0 - 2.0 * rgb); // cubic smoothing	

    return c.z * lerp(float3(1.0, 1.0, 1.0), rgb, c.y);
}

float3 mixN(float3 a, float3 b, float k)
{
    return sqrt(lerp(a * a, b * b, clamp(k, 0., 1.)));
}


