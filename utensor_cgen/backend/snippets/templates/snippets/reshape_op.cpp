{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new ReshapeOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endif %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}