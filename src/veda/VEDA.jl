module VEDA
    include("api.jl")

    #using .API

    include("enum.jl")
    include("error.jl")
    include("devices.jl")

    export VEModule, VEFunction, VEContext

    # Hardcoded linker path: from binutils-ve
    const nld_path = "/opt/nec/ve/bin/nld"

    # TODO: Find these at build time.

    mutable struct VEContext
        handle::VEDAcontext

        function VEContext(dev)
            r_handle = Ref{VEDAcontext}()
            mode = VEDA_CONTEXT_MODE_OMP
            if haskey(ENV, "VE_CONTEXT_MODE") && ENV["VE_CONTEXT_MODE"] == "SCALAR"
                mode = VEDA_CONTEXT_MODE_SCALAR
            end
            @check vedaCtxCreate(r_handle, mode, dev)
            ctx = new(r_handle[])
            finalizer(ctx) do ctx
                vedaCtxDestroy(ctx.handle)
            end
        end
    end

    mutable struct VEModule
        handle::VEDAmodule

        function VEModule(vso::String)
            r_handle = Ref{VEDAmodule}()
            @check vedaModuleLoad(r_handle, vso)
            mod = new(r_handle[])
            finalizer(mod) do mod
                vedaModuleUnload(mod.handle)
            end
            return mod
        end
    end

    function VEModule(obj::Base.CodeUnits{UInt8, String})
        vso = mktemp() do path_o, io_o
            write(io_o, obj)
            flush(io_o)
            vso = path_o*".vso"
            run(`$nld_path -shared -o $vso $path_o`)
            vso
        end
        mod = VEModule(vso)
        rm(vso)
        mod
    end

    mutable struct VEFunction
        handle::VEDAfunction
        mod::VEModule

        function VEFunction(mod, fname)
            r_handle = Ref{VEDAfunction}()
            @check vedaModuleGetFunction(r_handle, mod.handle, fname)
            new(r_handle[], mod)
        end
    end

    include("execution.jl")

    const pctx = Ref{VEContext}()

    ## VEDA device library
    const libcache = Base.WeakKeyDict{VEContext, VEModule}()

    function __init__()
        isempty(libveda) && return
        # limiting to one core for now, as we're not using OpenMP
        #ENV["VE_OMP_NUM_THREADS"] = 1
        # TODO: Do a lazy init
        vedaInit(0)
        ctx = VEContext(0)
        pctx[] = ctx
        vedaCtxSetCurrent(ctx.handle)
        #atexit(vedaExit)
    end
end # module
