{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{   
    {% if ref_count%}
    ctx.add({{output_vars[0]}}, "{{output}}", {{ref_count}});
    {% else %}
    ctx.add({{output_vars[0]}}, "{{output}}");
    {% endif %}
    ctx.push(new MinOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endif %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}
