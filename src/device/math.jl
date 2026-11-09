# math functionality

## trigonometric

@device_override Base.sin(x::Float64) = ccall("extern sin", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sin(x::Float32) = ccall("extern sinf", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sind(x) = sin(deg2rad(x))
@device_override Base.sinpi(x) = sin(x * pi)

@device_override Base.cos(x::Float64) = ccall("extern cos", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cos(x::Float32) = ccall("extern cosf", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cosd(x) = cos(deg2rad(x))
@device_override Base.cospi(x) = cos(x * pi)

@device_override Base.tan(x::Float64) = ccall("extern tan", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.tan(x::Float32) = ccall("extern tanf", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tand(x) = tan(deg2rad(x))

@device_override Base.sincos(x) = (sin(x), cos(x))  # TODO: dispatch to native `sincos`
@device_override Base.sincosd(x) = (sind(x), cosd(x))
@device_override Base.sincospi(x) = (sinpi(x), cospi(x))


## inverse trigonometric

@device_override Base.asin(x::Float64) = ccall("extern asin", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asin(x::Float32) = ccall("extern asinf", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.asind(x) = rad2deg(asin(x))

@device_override Base.acos(x::Float64) = ccall("extern acos", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acos(x::Float32) = ccall("extern acosf", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.acosd(x) = rad2deg(acos(x))

@device_override Base.atan(y::Float64) = ccall("extern atan", llvmcall, Cdouble, (Cdouble,), y)
@device_override Base.atan(y::Float32) = ccall("extern atanf", llvmcall, Cfloat, (Cfloat,), y)
@device_override Base.atand(y) = rad2deg(atan(y))

@device_override Base.atan(y::Float64, x::Float64) = ccall("extern atan2", llvmcall, Cdouble, (Cdouble, Cdouble), y, x)
@device_override Base.atan(y::Float32, x::Float32) = ccall("extern atan2f", llvmcall, Cfloat, (Cfloat, Cfloat), y, x)
@device_override Base.atand(y, x) = rad2deg(atan(y, x))


## hyperbolic

@device_override Base.sinh(x::Float64) = ccall("extern sinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sinh(x::Float32) = ccall("extern sinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.cosh(x::Float64) = ccall("extern cosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cosh(x::Float32) = ccall("extern coshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.tanh(x::Float64) = ccall("extern tanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.tanh(x::Float32) = ccall("extern tanhf", llvmcall, Cfloat, (Cfloat,), x)


## inverse hyperbolic

@device_override Base.asinh(x::Float64) = ccall("extern asinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asinh(x::Float32) = ccall("extern asinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.acosh(x::Float64) = ccall("extern acosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acosh(x::Float32) = ccall("extern acoshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atanh(x::Float64) = ccall("extern atanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.atanh(x::Float32) = ccall("extern atanhf", llvmcall, Cfloat, (Cfloat,), x)


## logarithmic

@device_override Base.log(x::Float64) = ccall("extern log", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log(x::Float32) = ccall("extern logf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log10(x::Float64) = ccall("extern log10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log10(x::Float32) = ccall("extern log10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log1p(x) = log(1 + x)

const _inv_log_2 = inv(log(2))
@device_override Base.log2(x::Float64) = log(x) * _inv_log_2  # TODO: dispatch to native `log2`
@device_override Base.log2(x::Float32) = log(x) * Float32(_inv_log_2)


## exponential

@device_override Base.exp(x::Float64) = ccall("extern exp", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp(x::Float32) = ccall("extern expf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.exp2(x::Float64) = ccall("extern exp2", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp2(x::Float32) = ccall("extern exp2f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.exp10(x::Float64) = ccall("extern exp10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp10(x::Float32) = ccall("extern exp10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.expm1(x::Float64) = ccall("extern expm1", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.expm1(x::Float32) = ccall("extern expm1f", llvmcall, Cfloat, (Cfloat,), x)


## roots and powers

@inline function rsqrt(x::Float64)
    y = 1.0 / Core.Intrinsics.sqrt_llvm(x)
    y += 0.5 * y * (1.0 - x * y * y)
    y += 0.5 * y * (1.0 - x * y * y)
    return y
end
@inline function rsqrt(x::Float32)
    y = 1.0f0 / Core.Intrinsics.sqrt_llvm(x)
    y += 0.5f0 * y * (1.0f0 - x * y * y)
    return y
end

@device_override @inline function Base.sqrt(x::Float64)
    y = 1.0 / Core.Intrinsics.sqrt_llvm(x)
    y += 0.5 * y * (1.0 - x * y * y)
    z = x * y
    z += 0.5 * z * (1.0 - z * y)
    return z
end
@device_override Base.sqrt(x::Float32) = convert(Float32, sqrt(convert(Float64, x)))

@device_override Base.FastMath.sqrt_fast(x) = sqrt(x)

@device_override Base.cbrt(x::Float64) = ccall("extern cbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cbrt(x::Float32) = ccall("extern cbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.:(^)(x::Float64, y::Float64) = ccall("extern pow", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.:(^)(x::Float32, y::Float32) = ccall("extern powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.:(^)(x, y::Integer) = x ^ convert(typeof(x), y)

@device_override Base.FastMath.pow_fast(x, y::Integer) = x ^ convert(typeof(x), y)

@device_override @inline function Base.literal_pow(::typeof(^), x, ::Val{p}) where {p}
    p <= -2 && return x ^ p
    p == -1 && return inv(x)
    p == 0 && return one(x)
    y = Base.literal_pow(^, x, Val(div(p, 2)))
    if isodd(p)
        return x * y * y
    else
        return y * y
    end
end
