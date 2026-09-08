# global state management

export context!, device!

mutable struct TaskLocalState
    device::VEDevice
    context::VEContext
    contexts::Vector{Union{Nothing, VEContext}}

    function TaskLocalState()
        dev = VEDevice(0)
        ctx = VEContext(0)
        vedaCtxSetCurrent(ctx.handle)
        contexts = Vector{Union{Nothing, VEContext}}(nothing, ndevices())
        contexts[1] = ctx
        new(dev, ctx, contexts)
    end
end

# initialized in __init__() in VEDA.jl
const state_ref = Ref{TaskLocalState}()

function context()
    return state_ref[].context
end

function context(dev::VEDevice)
    contexts = state_ref[].contexts
    devidx = deviceid(dev)+1
    if contexts[devidx] === nothing
        contexts[devidx] = VEContext(dev)
    end
    return contexts[devidx]::VEContext
end

function context!(ctx::VEContext)
    state = state_ref[]
    old_ctx = state.context
    if old_ctx != ctx
        vedaCtxSetCurrent(ctx.handle)
        dev = current_device()
        state.device = dev
        state.context = ctx
    end
    return old_ctx
end

@inline function context!(f::Function, ctx::VEContext)
    old_ctx = context!(ctx)
    try
        f()
    finally
        context!(old_ctx)
    end
end

function device()
    return state_ref[].device
end

function device!(dev::VEDevice)
    ctx = context(dev)
    vedaCtxSetCurrent(ctx.handle)
    state = state_ref[]
    state.device = dev
    state.context = ctx
    return dev
end

function device!(f::Function, dev::VEDevice)
    ctx = context(dev)
    context!(f, ctx)
    return dev
end

device!(dev::Integer) = device!(VEDevice(dev))
device!(f::Function, dev::Integer) = device!(f, VEDevice(dev))
