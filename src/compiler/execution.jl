# VE execution support

export @veda, vedaconvert, vefunction

## high-level @cuda interface

"""
    @veda [kwargs...] func(args...)

High-level interface for executing code on a Vector Engine. The `@veda` macro should prefix a call,
with `func` a callable function or object that should return nothing. It will be compiled to
a VE function upon first use, and to a certain extent arguments will be converted and
managed automatically using `vedaconvert`. Finally, a call to `vedacall` is
performed, scheduling a kernel launch on the current VEDA context.

Several keyword arguments are supported that influence the behavior of `@veda`.
- `launch`: whether to launch this kernel, defaults to `true`. If `false` the returned
  kernel object should be launched by calling it and passing arguments again.
- `dynamic`: use dynamic parallelism to launch device-side kernels, defaults to `false`.
- arguments that influence kernel compilation: see [`vefunction`](@ref) and
  [`dynamic_vefunction`](@ref)
- arguments that influence kernel launch: see [`VEDA.HostKernel`](@ref) and
  [`VEDA.DeviceKernel`](@ref)
"""
macro veda(ex...)
    # destructure the `@veda` expression
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @veda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs,
                     [:launch],
                     [:name, :global_hooks, :device, :kernel],
                     [:stream])
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key == :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @veda should be a constant value"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@veda with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel
    # regular, host-side kernel launch
    #
    # convert the function, its arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                local $kernel_f = $vedaconvert($f_var)
                local $kernel_args = map($vedaconvert, ($(var_exprs...),))
                local $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                local $kernel = $vefunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)
    return esc(code)
end


## host to device value conversion

struct Adaptor end

# convert host pointers to device pointers
Adapt.adapt_storage(to::Adaptor, p::VEPtr{T}) where {T} = reinterpret(Ptr{T}, p)

## Base.RefValue isn't VE compatible, so provide a compatible alternative
#struct VERefValue{T} <: Ref{T}
#  x::T
#end
#Base.getindex(r::VERefValue) = r.x
#Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = VERefValue(adapt(to, r[]))

#Adapt.adapt_structure(to::Adaptor, x::Base.RefValue{T}) where {T} =
#    Base.unsafe_convert(VERef{T}, x)

## broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
## avoid that by using a special kind of ref that knows about the boxed type.
#struct VERefType{T} <: Ref{DataType} end
#Base.getindex(r::VERefType{T}) where T = T
#Adapt.adapt_structure(to::Adaptor, r::Base.RefValue{<:Union{DataType,Type}}) = VERefType{r[]}()

"""
    vedaconvert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a device-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `VectorEngine.Adaptor` type.
"""
vedaconvert(arg) = adapt(Adaptor(), arg)


###################################

## abstract kernel functionality

abstract type AbstractKernel{F,TT} end


struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    #ctx::VEContext
    mod::VEModule
    fun::VEFunction
end

"""
    (::HostKernel)(args...; kwargs...)
    (::DeviceKernel)(args...; kwargs...)

Low-level interface to call a compiled kernel, passing GPU-compatible arguments in `args`.
For a higher-level interface, use [`@veda`](@ref).

The following keyword arguments are supported:
- `stream` (defaults to the default stream)
"""
AbstractKernel

@generated function call(kernel::AbstractKernel{F,TT}, args...; call_kwargs...) where {F,TT}
    sig = Base.signature_type(F, TT)
    args = (:F, (:( args[$i] ) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    #predicate = dt -> isghosttype(dt) || Core.Compiler.isconstType(dt)
    #to_pass = map(!predicate, sig.parameters)
    to_pass = map(!isghosttype, sig.parameters)
    call_t =                  Type[x[1] for x in zip(sig.parameters,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]
    @debug("call_t=$call_t")
    @debug("call_args=$call_args")
    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    # alternatively, allow `launch` with non-isbits arguments.
    # make sure the type is none of those passed on stack by VE.
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt) && !isstructtype(dt) && !(dt <: Base.RefValue)
            @debug("!isbitstype detected: argument $i $dt")
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        Base.@_inline_meta

        err, veargs = vecall(kernel, $call_tt, $(call_args...); call_kwargs...)
        return veargs
    end
end

using LLVM, LLVM.Interop # for the correct definition of isghosttype

struct llvm_Optional{T}
    value::T
    hasVal::Bool
end

function llvm_VPIntrinsic_isVPIntrinsic(ID)
    ccall((:_ZN4llvm11VPIntrinsic13isVPIntrinsicEj, LLVM.libllvm), Bool, (Cuint,), ID)
end

function llvm_VPIntrinsic_getFunctionalOpcodeForVP(ID)
    ccall((:_ZN4llvm11VPIntrinsic24getFunctionalOpcodeForVPEj, LLVM.libllvm), llvm_Optional{Cuint}, (Cuint,), ID)
end

function llvm_Instruction_setFast(this, B)
    ccall((:_ZN4llvm11Instruction7setFastEb, LLVM.libllvm), Cvoid, (LLVM.API.LLVMValueRef, Bool), this, B)
end

# from llvm/include/llvm/IR/Instruction.def
const map_to_llvmopcode = map(op -> (@eval LLVM.API.$(Symbol(:LLVM, op))), [
    # TERM_INST
    :Ret, :Br, :Switch, :IndirectBr, :Invoke, :Resume, :Unreachable, :CleanupRet, :CatchRet, :CatchSwitch, :CallBr,
    # UNARY_INST
    :FNeg,
    # BINARY_INST
    :Add, :FAdd, :Sub, :FSub, :Mul, :FMul, :UDiv, :SDiv, :FDiv, :URem, :SRem, :FRem, :Shl, :LShr, :AShr, :And, :Or, :Xor,
    # MEMORY_INST
    :Alloca, :Load, :Store, :GetElementPtr, :Fence, :AtomicCmpXchg, :AtomicRMW,
    # CAST_INST
    :Trunc, :ZExt, :SExt, :FPToUI, :FPToSI, :UIToFP, :SIToFP, :FPTrunc, :FPExt, :PtrToInt, :IntToPtr, :BitCast, :AddrSpaceCast,
    # FUNCLETPAD_INST
    :CleanupPad, :CatchPad,
    # OTHER_INST
    :ICmp, :FCmp, :PHI, :Call, :Select, :UserOp1, :UserOp2, :VAArg, :ExtractElement, :InsertElement, :ShuffleVector, :ExtractValue, :InsertValue, :LandingPad, :Freeze,
])

function is_fp_math_operator(inst::LLVM.Instruction)
    op = opcode(inst)
    if LLVM.API.LLVMIsAIntrinsicInst(inst) != C_NULL
        id = LLVM.API.LLVMGetIntrinsicID(called_value(inst)::LLVM.Function)
        if llvm_VPIntrinsic_isVPIntrinsic(id)
            op_opt = llvm_VPIntrinsic_getFunctionalOpcodeForVP(id)
            op = (op_opt.hasVal ? map_to_llvmopcode[op_opt.value] : LLVM.API.LLVMCall)
        end
    end

    if op in [LLVM.API.LLVMFNeg, LLVM.API.LLVMFAdd, LLVM.API.LLVMFSub, LLVM.API.LLVMFMul, LLVM.API.LLVMFDiv, LLVM.API.LLVMFRem, LLVM.API.LLVMFCmp]
        return true
    elseif op in [LLVM.API.LLVMPHI, LLVM.API.LLVMSelect, LLVM.API.LLVMCall]
        ty = LLVM.llvmtype(inst)
        while ty isa LLVM.ArrayType
            ty = eltype(ty)
        end
        if ty isa LLVM.VectorType
            ty = eltype(ty)
        end
        return (ty isa LLVM.FloatingPointType)
    else
        return false
    end
end

function force_fast_math!(mod::LLVM.Module)
    changed = false
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        if is_fp_math_operator(inst)
            llvm_Instruction_setFast(inst, true)
            changed = true
        end
    end
    return changed
end

function fast_loop_vectorize!(pm::LLVM.PassManager)
    rv_loop_vectorize!(pm)
    add!(pm, LLVM.ModulePass("ForceFastMath", force_fast_math!))
end

"""
    vefunction(f, tt=Tuple{}; kwargs...)

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object.

The following keyword arguments are supported:
- `name`: overrides the name that the kernel will have in the generated code
- `global_hooks`: specifies maps from global variable name to initializer hook

The output of this function is automatically cached, i.e. you can simply call `vefunction`
in a hot path without degrading performance. New code will be generated automatically, when
function definitions change, or when different types or keyword arguments are provided.
"""
function vefunction(f::Core.Function, tt::Type=Tuple{}; name=nothing, device=0,
                    global_hooks=NamedTuple(), kernel=true, kwargs...)
    source = FunctionSpec(f, tt, kernel, name)
    cache = get!(()->Dict{UInt,Any}(), vefunction_cache, device)
    #isa = default_isa(device)
    target = VECompilerTarget()
    params = VECompilerParams(device, global_hooks, fast_loop_vectorize!)
    job = CompilerJob(target, source, params)
    GPUCompiler.cached_compilation(cache, job, vefunction_compile, vefunction_link)::HostKernel{f,tt}
end

const vefunction_cache = Dict{UInt,Dict{UInt,Any}}()

function vefunction_compile(@nospecialize(job::CompilerJob))
    # compile
    method_instance, mi_meta = GPUCompiler.emit_julia(job)
    ir, ir_meta = GPUCompiler.emit_llvm(job, method_instance; ctx=JuliaContext())
    kernel = ir_meta.entry
    if haskey(ENV, "JULIA_VE_IR") && ENV["JULIA_VE_IR"] == "1"
        @show ir
    end

    obj, obj_meta = GPUCompiler.emit_asm(job, ir; format=LLVM.API.LLVMObjectFile)

    # find undefined globals and calculate sizes
    globals = map(gbl->Symbol(LLVM.name(gbl))=>llvmsize(eltype(llvmtype(gbl))),
                  filter(x->isextinit(x), collect(LLVM.globals(ir))))
    entry = LLVM.name(kernel)
    dispose(ir)

    return (;obj, entry, globals)
end

function vefunction_link(@nospecialize(job::CompilerJob), compiled)
    device = job.params.device
    global_hooks = job.params.global_hooks
    obj, entry, globals = compiled.obj, compiled.entry, compiled.globals

    # create executable and kernel
    obj = codeunits(obj)
    mod = VEModule(obj)
    fun = VEFunction(mod, entry)
    kernel = HostKernel{job.source.f,job.source.tt}(mod, fun)

    # initialize globals from hooks
    for gname in first.(globals)
        hook = nothing
        if haskey(default_global_hooks, gname)
            hook = default_global_hooks[gname]
        elseif haskey(global_hooks, gname)
            hook = global_hooks[gname]
        end
        if hook !== nothing
            @debug "Initializing global $gname"
            gbl = get_global(exe, gname)
            hook(gbl, mod, device)
        else
            @debug "Uninitialized global $gname"
            continue
        end
    end

    return kernel
end

function (kernel::HostKernel)(args...; kwargs...)
    call(kernel, map(vedaconvert, args)...; kwargs...)
end

@inline function vecall(kernel::HostKernel, tt, args...; kwargs...)
    err, veargs = VEDA.vecall(kernel.fun, tt, args...; kwargs...)
end
