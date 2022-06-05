export SlicedArrayLike
export getslicing, unsliced, arrayslicer
export AbstractSlicing, InnerDimsFirst, OuterDimsFirst, ArbitrarySlicing

# Temporary, until part of Compat:
@static if !isdefined(Base, :Slices)
    @static if isdefined(Compat, :EachRow)
        # JuliaLang/Compat.jl/pull/663
        using Compat: EachRow
        using Compat: EachCol
    else
        const EachRow{A,I} = Base.Generator{I,(typeof(eachrow(ones(Int32,2,2)).f).name.wrapper){A}}
        const EachCol{A,I} = Base.Generator{I,(typeof(eachcol(ones(Int32,2,2)).f).name.wrapper){A}}
        Base.parent(x::EachRow) = x.f.A
        Base.parent(x::EachCol) = x.f.A
    end    

    @static if isdefined(Compat, :EachSlice)
        # Not part of JuliaLang/Compat.jl/pull/663 yet, but hopefully will be:
        using Compat: EachSlice
    else
        const EachSlice{A,I} = Base.Generator{I,<:(typeof(eachslice(ones(Int32,2,2), dims = 1).f).name.wrapper){A}}
        Base.parent(x::EachSlice) = x.f.A
    end

    @static if isdefined(Compat, :AbstractSlices)
        # Not part of JuliaLang/Compat.jl/pull/663 yet, but hopefully will be:
        using Compat: AbstractSlices
    elseif isdefined(Base, :AbstractSlices)
        using Base: AbstractSlices
    else
        abstract type AbstractSlices{T,N} <: AbstractArray{T,N} end
    end
end


"""
    const SlicedArrayLike = Union{EachRow,EachCol,EachSlice,AbstractSlices}

Something that represents a sliced array. Should support

* `parent(sliced::SlicedArrayLike)` should return the original unsliced array
* [`getslicemap(sliced::SlicedArrayLike)`](@ref)
"""
@static if isdefined(Base, :Slices)
    const SlicedArrayLike = AbstractSlices
else
    const SlicedArrayLike = Union{EachRow,EachCol,EachSlice,AbstractSlices}
end


"""
    abstract type AbstractSlicing{N,M,L}

Abstract supertypes for schemas that specify how an L-dimensional array is
to be sliced into an N-dimensional array of M-dimensional arrays.
If all sliced dimensions are dropped from the outer array, then
`N + M == L`. If no sliced dimensions are dropped, then `N == L`.
"""
abstract type AbstractSlicing{N,M,L} end

"""
    getslicing(A::SlicedArrayLike)::AbstractSlicing

Returns the slicing sceme of A.
"""
function getslicing end

"""
    unsliced(A::SlicedArrayLike, slicing = getslicing(A))::AbstractArray

Returns a flattened/unsliced representation of `A`

Depending on `slicing` the order of dimensions of the resulting array may
differ from `parent(A)`.
"""
function unsliced end

"""
    arrayslicer(A::SlicedArrayLike, slicing = getslicing(A))::Function

Returns a function that slices arrays, using a sliced array type similar to
`typeof(A)` but possibly with a different slicing.
"""
function arrayslicer end


"""
    abstract type InnerDimsFirst{N,M,L} <: AbstractSlicing{N,M,L}

Represents a slicing of an L-dimensional array into M inner and
N outer dimensions, inner dimensions first.
"""
struct InnerDimsFirst{N,M,L} <: AbstractSlicing{N,M,L} end


"""
    abstract type OuterDimsFirst{N,M,L} <: AbstractSlicing{N,M,L}

Represents a slicing of an L-dimensional array into N outer and
M inner dimensions, outer dimensions first.
"""
struct OuterDimsFirst{N,M,L} <: AbstractSlicing{N,M,L} end


"""
    abstract type ArbitrarySlicing{N,M,L} <: AbstractSlicing{N,M,L}

Represents an arbitrary slicing of an L-dimensional array into
an N-dimensional array of M-dimensional arrays.
"""
struct ArbitrarySlicing{N,M,L,S<:NTuple{L,Union{Int,Colon}}} <: AbstractSlicing{N,M,L}
    slicemap::S
end

ArbitrarySlicing{N,M,L}(slicemap::S) where {N,M,L,S<:NTuple{L,Union{Int,Colon}}} = ArbitrarySlicing{N,M,L,S}(slicemap)

ArbitrarySlicing(::InnerDimsFirst{N,M,L}) where {N,M,L} = ArbitrarySlicing{N,M,L}(ntuple(i -> (i <= M ? Colon() : i + N - L), Val(L)))
ArbitrarySlicing(::OuterDimsFirst{N,M,L}) where {N,M,L} = ArbitrarySlicing{N,M,L}(ntuple(i -> (i < M ? i : Colon()), Val(L)))

Base.convert(::Type{ArbitrarySlicing}, slicing::ArbitrarySlicing) = slicing
Base.convert(::Type{ArbitrarySlicing}, slicing::AbstractSlicing) = ArbitrarySlicing(slicing)


Base.eachslice(A, slicing::AbstractSlicing) = eachslice(A, convert(ArbitrarySlicing, slicing))

_seaxis(::Base.OneTo) = Base.OneTo(1)
_seaxis(axis::AbstractArray{<:Integer}) = first(axis):first(axis)

@inline _outerdims(drop::Bool, dropped::Int) = ()
@inline _outerdims(drop::Bool, dropped::Int, ::Colon, dims::Vararg{Union{Int,Colon},N}) where N = (_outerdims(drop, dropped + drop, dims...)...,)
@inline _outerdims(drop::Bool, dropped::Int, i::Integer, dims::Vararg{Union{Int,Colon},N}) where N = (i + dropped, _outerdims(drop, dropped, dims...)...)

function Base.eachslice(A::AbstractArray{<:Any,N}, slicing::ArbitrarySlicing{N,M,N}) where {N,M}
    ax =  map((s,a) -> (s isa Colon ? _seaxis(a) : a), slicing.slicemap, axes(A))
    Slices(A, slicing.slicemap, ax)
end

function Base.eachslice(A::AbstractArray{<:Any,L}, slicing::ArbitrarySlicing{N,M,L}) where {N,M,L}
    ax =  map(i -> axes(A, i), _outerdims(true, 0, slicing.slicemap...))
    Slices(A, slicing.slicemap, ax)
end


function getslicing(A::Slices{<:Any,<:NTuple{L,Union{Int,Colon}},<:NTuple{N,<:Any}}) where {N,L}
    return ArbitrarySlicing{N,N-length(A.axes),L}(A.slicemap)
end


function unsliced(A::Slices, slicing::ArbitrarySlicing = getslicing(A))
    orig_slicing = getslicing(A)
    # Supporting this would require an optional permutedims(), causing loss of type stability:
    #!!!!!! Handle more cases!
    orig_slicing.slicemap == slicing.slicemap || throw(ArgumentError("Unslicing Slices with different slicmaps not supported"))
    return parent(A)
end

unsliced(A::Slices, slicing::AbstractSlicing) = unsliced(A, convert(ArbitrarySlicing, slicing))


arrayslicer(A::Slices, slicing::ArbitrarySlicing = getslicing(A)) = Base.Fix2(eachslice, convert(ArbitrarySlicing, slicing))
